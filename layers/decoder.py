from typing import Optional

import torch.functional as F
from torch import Tensor, nn

from layers.layer_norm import LayerNorm
from layers.series_decomp import SeriesDecomp
from utils.types import Activation


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        D_model: int,
        D_out: int,
        D_ff: Optional[int] = None,
        q_mva: Optional[int] = 25,
        dropout: Optional[float] = 0.1,
        activation: Optional[Activation] = 'relu',
    ):
        """Initializes the DecoderLayer module.

        Args:
            self_attention: Initial self-attention module.
            cross_attention: Next cross-attention module.
            D_model: Feature dimension.
            D_out: Output dimension.
            D_ff: FFNN feature dimension. Defaults to 4 * D_model.
            q_mva: Moving average window size for series decomposition.
            dropout: Dropout probability.
            activation: Nonlinearity used in FFNN.
        """
        super().__init__()

        D_ff = D_ff or 4 * D_model

        self.conv1 = nn.Conv1d(D_model, D_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(D_ff, D_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(q_mva)
        self.decomp2 = SeriesDecomp(q_mva)
        self.decomp3 = SeriesDecomp(q_mva)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            D_model,
            D_out,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False,
        )
        self.activation = F.relu if activation == 'relu' else F.gelu

        self.self_attention = self_attention
        self.cross_attention = cross_attention

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the DecoderLayer module.

        Args:
            x: (B, L, D)
            cross: (B, S, D)
            x_mask: Mask for self attention.
            cross_mask: Mask for cross attention.

        Returns:
            Decoder (seasonal) output, residual trend.
        """
        x0 = x
        x, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x0 + self.dropout(x)
        x, t1 = self.decomp1(x)

        x0 = x
        x, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x0 + self.dropout(x)
        x, t2 = self.decomp2(x)

        x0 = x
        x = self.dropout(self.conv1(x.transpose(-2, -1)))
        x = x0 + self.dropout(self.conv2(x).transpose(-2, -1))
        x, t3 = self.decomp3(x)

        residual_trend = t1 + t2 + t3
        residual_trend = self.projection(residual_trend.transpose(-2, -1)).transpose(
            -2, -1
        )
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(
        self,
        layers: list[DecoderLayer],
        norm_layer: Optional[LayerNorm] = None,
        projection=None,
    ):
        """Decoder module.

        Args:
            layers: DecoderLayer modules.
            norm_layer: Custom layer normalization class.
            projection: Projection layer.
        """
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.projection = projection

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
        trend: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the Decoder module.

        Each DecoderLayer is applied in sequence, with the output of the
        previous layer applied as input to the next layer.
        Finally, it applies a layer normalization and projection layer
        if specified.

        Args:
            x: Seasonal component of the input series. Shape = (B, L, D).
            cross: Seasonal component of encoder series. Shape = (B, S, D).
            x_mask: Mask for self-attention module.
            cross_mask: Mask for cross-attention module.
            trend: Trend component of the input series. Shape = (B, L, D).

        Returns:
            Decoder output, residual trend.
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask, cross_mask)
            trend = trend + residual_trend

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
