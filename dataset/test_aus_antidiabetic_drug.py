import numpy as np
import pytest
import torch

from dataset.aus_antidiabetic_drug import (
    Dataset_AusAntidiabeticDrug,
    aus_antidiabetic_drug_loaders,
)

BATCH_SIZE = 16
LEN_SEQ = 8
LEN_LABEL = 4
LEN_PRED = 8
NUM_PERIODS = 4


@pytest.mark.xfail
@pytest.mark.parametrize(
    'len_seq, len_label, len_pred', [(8, 4, 4), (4, 4, 4), (4, 4, 8)]
)
def test_bad_init(len_seq, len_label, len_pred):
    Dataset_AusAntidiabeticDrug(len_seq, len_label, len_pred)


def test_scale():
    dataset = Dataset_AusAntidiabeticDrug(8, 4, 8)
    assert np.allclose(np.mean(dataset.data_x, axis=0), 0.0, atol=1e-3)
    assert np.allclose(np.var(dataset.data_x, axis=0), 1.0, atol=1e-3)


@pytest.fixture(scope='module')
def dataloaders():
    return aus_antidiabetic_drug_loaders(0.8, LEN_SEQ, LEN_LABEL, LEN_PRED, BATCH_SIZE)


def test_loads_data_in_batches(dataloaders):
    def check(seq_x, seq_x_mark, seq_y, seq_y_mark):
        assert seq_x.shape == (BATCH_SIZE, LEN_SEQ, 1)
        assert seq_x_mark.shape == (BATCH_SIZE, LEN_SEQ, NUM_PERIODS)
        assert seq_y.shape == (BATCH_SIZE, LEN_LABEL + LEN_PRED, 1)
        assert seq_y_mark.shape == (BATCH_SIZE, LEN_LABEL + LEN_PRED, NUM_PERIODS)

    train_loader, test_loader = dataloaders
    for batch in train_loader:
        check(*batch)
    for i, (seq_x, seq_x_mark, seq_y, seq_y_mark) in enumerate(test_loader):
        if i == len(test_loader) - 1:
            assert seq_x.shape[1:] == (LEN_SEQ, 1)
            assert seq_x_mark.shape[1:] == (LEN_SEQ, NUM_PERIODS)
            assert seq_y.shape[1:] == (LEN_LABEL + LEN_PRED, 1)
            assert seq_y_mark.shape[1:] == (LEN_LABEL + LEN_PRED, NUM_PERIODS)
            break
        check(*batch)
