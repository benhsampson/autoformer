from torch import Tensor, nn

from layers.mva import MVA


class SeriesDecomp(nn.Module):
    def __init__(self, q_mva: int):
        """Class for decomposing a time series into its trend and seasonal components.

        Args:
            q_mva: Window size for the moving average.
        """
        super().__init__()
        self.mva = MVA(q_mva)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trend = self.mva(x)
        seasonal = x - trend
        return seasonal, trend
