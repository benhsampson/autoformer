from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.decoder import Decoder, DecoderLayer
from layers.embedding import DataEmbedding, Frequency
from layers.encoder import Encoder, EncoderLayer
from layers.layer_norm import LayerNorm
from layers.series_decomp import SeriesDecomp
from utils.types import Activation


@dataclass
class AutoformerConfig:
    D_model: int  # feature dimension
    D_ff: int  # inner-feature dimension for FFNNs of encoder and decoder
    c_autocorrelation: (
        float
    )  # coefficient for auto-correlation, recall that k = floor(c * log L)
    num_heads: int
    q_mva: int  # window size for moving average in series decomposition
    dropout: float  # dropout probability
    activation: Activation  # activation function for FFNNs of encoder and decoder
    highest_freq: Frequency  # highest frequency for temporal encoding
    num_encoder_layers: int
    num_decoder_layers: int
    enc_in: int  # feature dimension of encoder
    dec_in: int  # feature dimension of decoder
    L_max: int  # maximum sequence length, used for positional encoding
    D_out: int  # output dimension
    len_seq: int  # sequence length
    len_pred: int  # the last len_pred points are predicted
    len_label: int  # the last len_label points are unseen


class Autoformer(nn.Module):
    def __init__(self, conf: AutoformerConfig):
        super().__init__()

        self.decomp = SeriesDecomp(conf.q_mva)

        self.enc_embedding = DataEmbedding(
            conf.enc_in,
            conf.D_model,
            conf.L_max,
            embed_positions=False,
            highest_freq='h',
            dropout=conf.dropout,
        )
        self.dec_embedding = DataEmbedding(
            conf.dec_in,
            conf.D_model,
            conf.L_max,
            embed_positions=False,
            highest_freq='h',
            dropout=conf.dropout,
        )

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(conf.c_autocorrelation),
                        conf.D_model,
                        conf.num_heads,
                    ),
                    conf.D_model,
                    conf.D_ff,
                    conf.q_mva,
                    conf.dropout,
                    conf.activation,
                )
                for _ in range(conf.num_encoder_layers)
            ],
            norm_layer=LayerNorm(conf.D_model),
        )

        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(conf.c_autocorrelation),
                        conf.D_model,
                        conf.num_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(conf.c_autocorrelation),
                        conf.D_model,
                        conf.num_heads,
                    ),
                    conf.D_model,
                    conf.D_out,
                    conf.D_ff,
                    conf.q_mva,
                    conf.dropout,
                    conf.activation,
                )
                for _ in range(conf.num_decoder_layers)
            ],
            norm_layer=LayerNorm(conf.D_model),
            projection=nn.Linear(conf.D_model, conf.D_out),
        )

        self.len_pred = conf.len_pred
        self.len_label = conf.len_label

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
        enc_self_mask: Optional[Tensor] = None,
        dec_self_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass of the Autoformer.

        Args:
            x_enc: Encoder input.
            x_mark_enc: Encoder timestamp input.
            x_dec: Decoder input.
            x_mark_dec: Decoder timestamp input.
            enc_self_mask: Encoder self-attention mask.
            dec_self_mask: Decoder self-attention mask.

        Returns:
            Autoformer output, encoder attention weights.
        """
        # run encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.len_pred, 1)
        zeros = torch.zeros_like(mean)
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.len_label :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.len_label :], zeros], dim=1)

        # run decoder
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder.forward(
            dec_in,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=enc_self_mask,
            trend=trend_init,
        )

        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.len_pred :, :], enc_attns
