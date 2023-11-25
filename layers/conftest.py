import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope='module')
def aus_antidiabetic_drug_data():
    df = pd.read_csv('data/AusAntidiabeticDrug.csv')
    x = df['y'].values.astype(np.float32)
    x = torch.as_tensor(x).view(1, -1, 1)
    return x
