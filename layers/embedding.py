from math import log
from typing import Literal

import torch
from torch import Tensor, nn

PERIODS = ['month', 'day', 'weekday', 'hour', 'minute']
PERIOD2IX = {p: i for i, p in enumerate(PERIODS)}

Frequency = Literal['t', 'h']


class TokenEmbedding(nn.Module):
    def __init__(self, C_in: int, D_model: int):
        super().__init__()

        self.token_conv = nn.Conv1d(
            C_in, D_model, kernel_size=3, padding=1, padding_mode='circular', bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(
        self,
        x: Tensor,  # (B, L, C)
    ) -> Tensor:  # (B, L, D)
        x = self.token_conv(x.transpose(-2, -1)).transpose(-2, -1)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, D_model: int, L_max: int):
        super().__init__()

        pe = torch.zeros(L_max, D_model)
        pe.requires_grad = False

        position = torch.arange(0, L_max).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_model, 2) * -(log(10000.0) / D_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # Buffers are tensors which do not require gradients
        # and are thus not registered as parameters.
        # They can be loaded inside the state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        self.pe: Tensor
        return self.pe[:, : x.shape[1]]


class TemporalEmbedding(nn.Module):
    def __init__(self, D_model: int, max_freq: Frequency):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        if max_freq == 't':
            self.minute_embd = nn.Embedding(minute_size, D_model)
        self.hour_embd = nn.Embedding(hour_size, D_model)
        self.weekday_embd = nn.Embedding(weekday_size, D_model)
        self.day_embd = nn.Embedding(day_size, D_model)
        self.month_embd = nn.Embedding(month_size, D_model)

    def forward(
        self,
        x_mark: Tensor,  # (B, L, P) where P = len(PERIODS)
    ):
        minute_x = (
            self.minute_embd(x_mark[..., PERIOD2IX['minute']])
            if hasattr(self, 'minute_embd')
            else 0.0
        )
        hour_x = self.hour_embd(x_mark[..., PERIOD2IX['hour']])
        weekday_x = self.weekday_embd(x_mark[..., PERIOD2IX['weekday']])
        day_x = self.day_embd(x_mark[..., PERIOD2IX['day']])
        month_x = self.month_embd(x_mark[..., PERIOD2IX['month']])

        return minute_x + hour_x + weekday_x + day_x + month_x


class DataEmbedding(nn.Module):
    def __init__(
        self,
        C_in: int,
        D_model: int,
        L_max: int,
        embed_positions: bool = True,
        highest_freq: Frequency = 'h',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embd = TokenEmbedding(C_in, D_model)
        self.pos_embd = PositionalEmbedding(D_model, L_max)
        self.temp_embd = TemporalEmbedding(D_model, highest_freq)
        self.dropout = nn.Dropout(dropout)

        self.embed_positions = embed_positions

    def forward(self, x: Tensor, x_mark: Tensor):
        x = self.token_embd(x) + self.temp_embd(x_mark)
        if self.embed_positions:
            x += self.pos_embd(x)
        return self.dropout(x)
