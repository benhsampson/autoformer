from typing import Optional

import torch.functional as F
from torch import Tensor, nn

from layers.layer_norm import LayerNorm
from layers.series_decomp import SeriesDecomp
from utils.types import Activation


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        D_model: int,
        D_ff: Optional[int] = None,
        q_mva: Optional[int] = 25,
        dropout: Optional[float] = 0.2,
        activation: Optional[Activation] = 'relu',
    ):
        """Initializes the EncoderLayer module.

        Args:
            attention: Attention module to use. Must be a nn.Module with a forward method
                that takes in queries, keys, values, and attn_mask as arguments, and
                returns a tuple of (output, attn).
            D_model: Feature dimension.
            D_ff: Feed-forward neural net (FFNN) feature dimension. Defaults to D_model * 4.
            q_mva: Window size for moving average in series decomposition.
            dropout: Dropout probability.
            activation: Nonlinearity for FFNN.
        """
        super().__init__()

        D_ff = D_ff or D_model * 4

        self.dropout = nn.Dropout(dropout)
        self.decomp1 = SeriesDecomp(q_mva)
        self.decomp2 = SeriesDecomp(q_mva)
        self.conv1 = nn.Conv1d(D_model, D_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(D_ff, D_model, kernel_size=1, bias=False)
        self.activation = F.relu if activation == 'relu' else F.gelu

        self.attention = attention

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the EncoderLayer module.

        Fully self-attentive with residual connections and dropout.

        Eliminates the seasonal component of the input series

        Args:
            x: (B, L, T)
            attn_mask: Unused.

        Returns: Encoder output, attention weights.
        """
        x0 = x
        x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x0 + self.dropout(x)

        x, _ = self.decomp1(x)

        x0 = x
        x = self.dropout(self.activation(self.conv1(x.transpose(-2, -1))))
        x = x0 + self.dropout(self.conv2(x).transpose(-2, -1))

        x, _ = self.decomp2(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(
        self,
        enc_layers: list[EncoderLayer],
        conv_layers: Optional[list[nn.Module]] = None,
        norm_layer: Optional[LayerNorm] = None,
    ):
        """Initializes Encoder module.

        Args:
            enc_layers: EncoderLayers to use.
            conv_layers: Convolutional layers. The i'th conv_layer is applied directly after the
                i'th attn_layer. If conv_layers is not None, ensure len(conv_layers) == len(enc_layers) - 1.
            norm_layer: Custom LayerNorm class for seasonal data.
        """
        super().__init__()

        if conv_layers is not None:
            assert len(conv_layers) == len(enc_layers) - 1

        self.enc_layers = nn.ModuleList(enc_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm_layer = norm_layer

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass of the Encoder module.

        Each EncoderLayer is applied in sequence, with the output of the previous layer
        passed into the next layer as input.

        Args:
            x: (B, L, D)
            attn_mask: Passed into each EncoderLayer.

        Returns:
            Encoder output, attention weights from each layer in enc_layers.
        """
        attns = []
        if self.conv_layers is not None:
            # ? untested
            for enc_layer, conv_layer in zip(self.enc_layers, self.conv_layers):
                x, attn = enc_layer(x)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.enc_layers[-1](x)
            attns.append(attn)
        else:
            for enc_layer in self.enc_layers:
                x, attn = enc_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x, attns
