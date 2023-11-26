from enum import Enum
from typing import Annotated, Optional

import typer
from torch import nn, optim

from autoformer import Autoformer, AutoformerConfig
from dataset.aus_antidiabetic_drug import aus_antidiabetic_drug_loaders
from layers.embedding import Frequency
from lstm import LSTM_TimeSeriesModel
from train import TrainArguments, train


class Model(str, Enum):
    lstm = 'lstm'
    autoformer = 'autoformer'


class Data(str, Enum):
    aus_antidiabetic_drug = 'aus_antidiabetic_drug'


class Task(str, Enum):
    multi_predict_multi = 'M'
    uni_predict_uni = 'S'
    multi_predict_uni = 'MS'


class Activation(str, Enum):
    relu = 'relu'
    gelu = 'gelu'


class Loss(str, Enum):
    mse = 'mse'


class Optimizer(str, Enum):
    adam = 'adam'


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    # ------------------- BASIC CONFIG ------------------- #
    is_training: Annotated[bool, typer.Option()] = True,
    model: Annotated[Model, typer.Option()] = Model.lstm,
    print_every_num_batches: Annotated[int, typer.Option()] = 100,
    # ------------------- DATA LOADER ------------------- #
    data: Annotated[Data, typer.Option()] = Data.aus_antidiabetic_drug,
    pct_train: Annotated[float, typer.Option()] = 0.8,
    task: Annotated[Task, typer.Option()] = Task.multi_predict_multi,
    target: Annotated[
        Optional[int], typer.Option(help='Target feature (required if S or MS task)')
    ] = None,
    highest_freq: Annotated[Frequency, typer.Option()] = 'h',
    checkpoints_dir: Annotated[str, typer.Option()] = 'checkpoints',
    # ------------------- FORECASTING TASK ------------------- #
    len_seq: Annotated[int, typer.Option()] = 96,
    len_label: Annotated[int, typer.Option()] = 48,
    len_pred: Annotated[int, typer.Option()] = 96,
    # ------------------- MODEL HYPERPARAMETERS ------------------- #
    # Autoformer
    enc_in: Annotated[int, typer.Option()] = 8,
    dec_in: Annotated[int, typer.Option()] = 8,
    C_out: Annotated[int, typer.Option()] = 8,
    D_model: Annotated[int, typer.Option()] = 512,
    num_heads: Annotated[int, typer.Option()] = 8,
    enc_layers: Annotated[int, typer.Option()] = 2,
    dec_layers: Annotated[int, typer.Option()] = 1,
    D_ff: Annotated[int, typer.Option()] = 2048,
    D_out: Annotated[int, typer.Option()] = 8,
    q_mva: Annotated[int, typer.Option()] = 25,
    c_autocorrelation: Annotated[int, typer.Option()] = 1,
    dropout: Annotated[float, typer.Option()] = 0.1,
    activation: Annotated[Activation, typer.Option()] = Activation.gelu,
    # LSTM
    hidden_size: Annotated[int, typer.Option()] = 32,
    # ------------------- OPTIMIZATION ------------------- #
    num_workers: Annotated[int, typer.Option()] = 8,
    train_epochs: Annotated[int, typer.Option()] = 10,
    batch_size: Annotated[int, typer.Option()] = 32,
    patience: Annotated[int, typer.Option()] = 3,
    delta: Annotated[float, typer.Option()] = 0.0,
    learning_rate: Annotated[float, typer.Option()] = 1e-4,
    loss: Annotated[Loss, typer.Option()] = Loss.mse,
    optimizer: Annotated[Optimizer, typer.Option()] = Optimizer.adam,
):
    assert not (
        task in {Task.multi_predict_uni, Task.uni_predict_uni} and target is None
    )

    if is_training:
        loader_dict = {
            Data.aus_antidiabetic_drug: aus_antidiabetic_drug_loaders(
                pct_train, len_seq, len_label, len_pred, batch_size
            )
        }
        train_loader, test_loader = loader_dict[data]
        model_dict = {
            Model.lstm: LSTM_TimeSeriesModel(
                input_size=dec_in,
                hidden_size=hidden_size,
                len_label=len_label,
                len_pred=len_pred,
            ),
            Model.autoformer: Autoformer(
                conf=AutoformerConfig(
                    D_model=D_model,
                    D_ff=D_ff,
                    c_autocorrelation=c_autocorrelation,
                    num_heads=num_heads,
                    q_mva=q_mva,
                    dropout=dropout,
                    activation=activation,
                    highest_freq=highest_freq,
                    num_encoder_layers=enc_layers,
                    num_decoder_layers=dec_layers,
                    L_max=10000,
                    D_out=D_out,
                    len_seq=len_seq,
                    len_pred=len_pred,
                    len_label=len_label,
                )
            ),
        }
        model_instance = model_dict[model]
        criterion_dict = {Loss.mse: nn.MSELoss()}
        # TODO: scheduler
        optimizer_dict = {
            Optimizer.adam: optim.Adam(model_instance.parameters(), lr=learning_rate)
        }

        train(
            TrainArguments(
                len_seq=len_seq,
                len_label=len_label,
                len_pred=len_pred,
                train_loader=train_loader,
                test_loader=test_loader,
                model=model_instance,
                criterion=criterion_dict[loss],
                optimizer=optimizer_dict[optimizer],
                train_epochs=train_epochs,
                checkpoints_dir=checkpoints_dir,
                patience=patience,
                delta=delta,
                verbose=True,
                print_every_num_batches=print_every_num_batches,
            )
        )


if __name__ == '__main__':
    app()
