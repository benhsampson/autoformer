from math import floor, log
from typing import Optional

import torch
from torch import Tensor, nn


class AutoCorrelation(nn.Module):
    """Multihead attention module with autocorrelation."""

    def __init__(self, c: float):
        """Initializes the auto-correlation module.

        Args:
            c: Parameter to use for the top-k selection where k = floor(c * log L).
        """
        super().__init__()
        self.c = c

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Attention mechanism using auto-correlation.

        Auto-correlation discovered periodic dependencies
        by aggregating the top-k sub-series with the highest
        auto-correlation.

        A sub-series is the original time series shifted to
        the left by a delay where elements shifted beyond
        the first position are reinserted at the end.

        Attention is calculated as the auto-correlation between
        the queries and the keys.

        Values are calculated as the sum of the top-k sub-series
        weighted by the softmax of their auto-correlation, so
        that they sum to 1 along L (the time dimension).

        We use the Fast Fourier Transform (FTT) to compute all
        the L auto-correlations in O(L) time. So computing the
        actual values takes O(L log L) time.

        Queries have length L, and keys and values have length S.
        If S < L, we right-pad keys and values with 0s, else
        if S > L, we truncate keys and values to length L.

        Args:
            queries: (B, L, nH, dH)
            keys: (B, S, nH, dH)
            values: (B, S, nH, dH')

        Returns:
            The values weighted by the attention weights (B, L, nH, dH'), and
            the attentions (B, L, nH, dH).
        """
        q, k, v = queries, keys, values

        B, L, nH, dH = q.shape
        _, S, _, _ = k.shape

        assert B == k.shape[0] == v.shape[0]
        assert S == v.shape[1]
        assert nH == k.shape[2] == v.shape[2]
        assert dH == k.shape[3]

        # Make q, k, v the same length L
        if S < L:
            zeros = torch.zeros((B, L - S, nH, dH))
            k = torch.cat([k, zeros], dim=1)
            v = torch.cat([v, zeros], dim=1)
        else:
            k = k[:, :L]
            v = v[:, :L]

        # Compute attentions (auto-correlations)
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        tmp_corr = q_fft * k_fft.conj()
        corr = torch.fft.irfft(tmp_corr, n=L, dim=1)

        top_k = floor(self.c * log(L))

        weights, delays = torch.topk(
            corr, top_k, dim=1
        )  # (B, k, nH, dH), (B, k, nH, dH)
        weights = torch.softmax(weights, dim=1)

        # Repeat the values twice so that elements shifted
        # beyond the 1st position reappear at the end
        tmp_values = v.repeat(1, 2, 1, 1)  # (B, 2L, nH, dH)
        init_index = torch.arange(L).view(1, L, 1, 1).repeat(B, 1, nH, dH)
        v_new = torch.zeros_like(v)

        for i in range(top_k):
            tmp_delay = init_index + delays[:, i].unsqueeze(1)
            pattern = torch.gather(tmp_values, dim=1, index=tmp_delay)
            v_new += pattern * weights[:, i].view(B, 1, nH, dH)

        return v_new, corr


class AutoCorrelationLayer(nn.Module):
    def __init__(self, attention: AutoCorrelation, D_model: int, n_heads: int):
        """Thin wrapper of the AutoCorrelation class.

        Args:
            attention: AutoCorrelation class.
            D_model: Feature dimension for the model.
            n_heads: Number of heads.
        """
        super().__init__()

        dQK = D_model // n_heads
        dV = D_model // n_heads

        self.query_projection = nn.Linear(D_model, dQK * n_heads)
        self.key_projection = nn.Linear(D_model, dQK * n_heads)
        self.value_projection = nn.Linear(D_model, dV * n_heads)
        self.out_projection = nn.Linear(dV * n_heads, D_model)

        self.attention = attention
        self.n_heads = n_heads

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the auto-correlation layer.

        Decomposes the queries, keys, and values into `num_heads`.
        Does auto-correlation...
        Re-projects the values into the original feature dimension.

        Args:
            queries: (B, L, D)
            keys: (B, S, D)
            values: (B, S, D)
            attn_mask: Unused.

        Returns:
            The auto-correlation output and the attention weights.
        """
        B, L, D = queries.shape
        _, S, _ = keys.shape
        nH = self.n_heads

        assert B == keys.shape[0] == values.shape[0]
        assert S == values.shape[1]
        assert D == keys.shape[2] == values.shape[2]

        queries = self.query_projection(queries).view(B, L, nH, -1)
        keys = self.key_projection(keys).view(B, S, nH, -1)
        values = self.value_projection(values).view(B, S, nH, -1)

        out, attn = self.attention(queries, keys, values)
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn
