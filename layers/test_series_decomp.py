import pytest
import torch

from layers.series_decomp import SeriesDecomp


@pytest.mark.parametrize('q', range(5, 50, 5))
def test_series_decomp(aus_antidiabetic_drug_data, q):
    x = aus_antidiabetic_drug_data
    sd = SeriesDecomp(q)
    x_seasonal, _ = sd(x)
    x_seasonal_mean = torch.mean(x_seasonal, dim=1)
    assert x_seasonal_mean == pytest.approx(0, abs=0.1)
