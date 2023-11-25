import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

from layers.embedding import PERIODS


class Dataset_AusAntidiabeticDrug(Dataset):
    def __init__(
        self,
        len_seq: int,
        len_label: int,
        len_pred: int,
    ):
        super().__init__()

        # the complete sequence will look something like this:
        # \\\\\\\\\\\\\\\\\\\\\\\XXXXXXXXXXXX/////////////////////
        # where \ is the input sequence of length len_seq
        # and   / is the output sequence of length len_pred,
        # and   X means it is shared between the input and output sequences.
        # The length of the shared part is len_label

        assert (
            len_label < len_seq
        ), 'at least one element should be exclusive to the input part'
        assert (
            len_label < len_pred
        ), 'at least one element should be exclusive to the output part'

        df_raw = pd.read_csv('data/AusAntidiabeticDrug.csv')

        # decompose date into periods
        df = df_raw.copy()
        df.loc[:, 'date'] = pd.to_datetime(df['ds'])
        dt = df['date'].dt
        df.loc[:, 'month'] = dt.month
        df.loc[:, 'day'] = dt.day
        df.loc[:, 'weekday'] = dt.weekday
        df.loc[:, 'hour'] = dt.hour
        df.drop(columns=['ds', 'date'], inplace=True)

        # split into x and x_mark
        df_x = df[['y']]
        data_x = df_x.values.astype(np.float32)
        df_x_mark = df.drop(columns=['y'])
        data_x_mark = df_x_mark[[p for p in PERIODS if p in df_x_mark.columns]].values

        # scale data to 0 mean and unit variance
        scaler = StandardScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)

        self.data_x = data_x
        self.data_x_mark = data_x_mark

        self.len_seq = len_seq
        self.len_label = len_label
        self.len_pred = len_pred

    def __len__(self):
        # let N be the total size of the time series
        # and M be the sequence size (input and output)
        # then the number of sequences is N - M + 1
        return len(self.data_x) - self.len_seq - self.len_pred + 1

    def __getitem__(self, index: int):
        """Gets a single item by index.

        Args:
            index: Index to retrieve.

        Returns:
            (seq_s, seq_s_mark, seq_r, seq_r_mark) where
                seq_s: Input sequence. Shape (len_seq, C).
                seq_s_mark: Input sequence marks. Shape (len_seq, P).
                seq_r: Output sequence. Shape (len_pred, C).
                seq_r_mark: Output sequence marks. Shape (len_label + len_pred, P).
        """
        x_begin = index
        x_end = x_begin + self.len_seq
        y_begin = x_end - self.len_label
        y_end = y_begin + self.len_label + self.len_pred

        seq_x = self.data_x[x_begin:x_end]
        seq_x_mark = self.data_x_mark[x_begin:x_end]

        seq_y = self.data_x[y_begin:y_end]
        seq_y_mark = self.data_x_mark[y_begin:y_end]

        return seq_x, seq_x_mark, seq_y, seq_y_mark


def aus_antidiabetic_drug_loaders(
    pct_train: float, len_seq: int, len_label: int, len_pred: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    assert 0.0 < pct_train < 1.0, 'invalid pct_train value, expected 0 < pct_train < 1'

    ds = Dataset_AusAntidiabeticDrug(len_seq, len_label, len_pred)

    train_size = int(pct_train * len(ds))
    test_size = len(ds) - train_size

    assert train_size > 0, 'train set must be non-empty'
    assert test_size > 0, 'test set must be non-empty'

    train, test = random_split(ds, [train_size, test_size])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, drop_last=False
    )
    # TODO: inference

    assert train_size > 0, 'train set must be non-empty'

    return train_loader, test_loader


# TODO: visualize
