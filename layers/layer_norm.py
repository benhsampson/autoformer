import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Specially designed layer norm for dealing with seasonal data."""

    def __init__(self, D: int):
        """Initializes the internal module state.

        Args:
            D: Feature dimension.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(D)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Computes the layer-normalized output.

        Initially, the layer is normalized across the feature dimension
        using the PyTorch LayerNorm class. However, given that `x` is a
        seasonal time series, we want its mean to be zero across the
        time dimension L. So we subtract the current mean to make it 0.

        Args:
            x: (B, L, D)

        Returns:
            Normalized time series.
        """
        x_hat = self.layer_norm(x)
        bias = torch.mean(x_hat, dim=1, keepdim=True)
        return x_hat - bias
