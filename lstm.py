from torch import Tensor, nn


class LSTM_TimeSeriesModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, len_label: int, len_pred: int
    ):
        super().__init__()

        self.model = nn.ModuleDict(
            {
                'lstm': nn.LSTM(input_size, hidden_size, batch_first=True),
                'linear': nn.Linear(hidden_size, 1),
            }
        )

        self.len_label = len_label
        self.len_pred = len_pred

    def forward(self, x: Tensor, *_) -> tuple[Tensor, None]:
        y_pred, _ = self.model['lstm'](x[:, self.len_label :, :])
        y_pred = self.model['linear'](y_pred)
        y_pred = y_pred[:, -self.len_pred :, :]
        return y_pred, None
