from math import ceil, floor

import torch
from torch import Tensor, nn


class MVA(nn.Module):
    def __init__(self, q: int):
        """Initializes the MVA module.

        Args:
            q: Window size.
        """
        super().__init__()
        self.q = q
        self.mva = nn.AvgPool1d(kernel_size=q, stride=1)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Computes the moving average of `x`.

        `x` is left-padded with the first value and right-padded with the last value.
        The moving average is computed with a kernel size of `q`.

        Args:
            x: Input tensor of shape (B, L, C).

        Returns:
            The moving average of `x` with same shape (B, L, C).
        """
        left_pad = x[:, 0:1].repeat(1, floor((self.q - 1) / 2), 1)
        right_pad = x[:, -1:].repeat(1, ceil((self.q - 1) / 2), 1)
        x_padded = torch.cat((left_pad, x, right_pad), dim=1)
        x_trend = self.mva(x_padded.transpose(-2, -1)).transpose(-2, -1)
        return x_trend
