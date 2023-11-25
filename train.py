import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from icecream import ic
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from early_stopping import EarlyStopping


@dataclass
class TrainArguments:
    len_seq: int
    len_label: int
    len_pred: int
    train_loader: DataLoader
    test_loader: DataLoader

    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer

    train_epochs: int

    checkpoints_dir: str  # path to folder to save checkpoints
    patience: int  # early stopping patience
    delta: float  # early stopping delta

    verbose: bool
    print_every_num_batches: int


def predict(
    model: nn.Module,
    batch_x: Tensor,  # (B, len_x, C)
    batch_x_mark: Tensor,  # (B, len_x, P)
    batch_y: Tensor,  # (B, len_y, C)
    batch_y_mark: Tensor,  # (B, len_y, P)
    len_label: int,
) -> tuple[Tensor, Tensor]:
    B, len_x, C = batch_x.shape
    _, len_y, _ = batch_y.shape
    _, _, P = batch_x_mark.shape

    assert B == batch_y.shape[0] == batch_x_mark.shape[0] == batch_y_mark.shape[0]
    assert C == batch_y.shape[2]
    assert len_x == batch_x_mark.shape[1]
    assert len_y == batch_y_mark.shape[1]
    assert P == batch_y_mark.shape[2]

    zeros = torch.zeros((B, len_y, C))
    dec_in = torch.cat(
        [batch_y[:, :len_label, :], zeros], dim=1
    )  # (B, len_label + len_pred, C)

    batch_y_pred, _ = model(dec_in, batch_x_mark, batch_y, batch_y_mark)
    batch_y_pred = batch_y_pred[:, -len_y:, :]  # (B, len_pred, C)
    batch_y_true = batch_y[:, len_label:, :]

    return batch_y_pred, batch_y_true


@torch.no_grad()
def eval(
    test_loader: DataLoader, model: nn.Module, criterion: nn.Module, len_label: int
) -> float:
    model.eval()

    losses = []

    for batch_x, batch_x_mark, batch_y, batch_y_mark in test_loader:
        batch_y_pred, batch_y_true = predict(
            model, batch_x, batch_x_mark, batch_y, batch_y_mark, len_label
        )
        batch_y_pred = batch_y_pred.detach()
        batch_y_true = batch_y_true.detach()
        loss = criterion(batch_y_pred, batch_y_true)
        losses.append(loss.item())

    loss_avg = np.mean(losses)

    model.train()

    return loss_avg


def train(train_arguments: TrainArguments):
    args = train_arguments

    checkpoints_path = Path(args.checkpoints_dir)
    if not checkpoints_path.exists():
        checkpoints_path.mkdir()

    model, criterion, optimizer = args.model, args.criterion, args.optimizer

    # TODO: learning rate scheduler

    early_stopping = EarlyStopping(
        model, optimizer, args.checkpoints_dir, args.patience, args.delta, args.verbose
    )

    time_before = time.time()

    len_train_loader = len(args.train_loader)
    assert len_train_loader > 0

    for epoch in range(args.train_epochs):
        train_losses = []
        iter_count = 0

        epoch_time = time.time()

        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(
            args.train_loader
        ):
            iter_count += 1

            optimizer.zero_grad()

            batch_y_pred, batch_y_true = predict(
                model, batch_x, batch_x_mark, batch_y, batch_y_mark, args.len_label
            )

            loss = criterion(batch_y_pred, batch_y_true)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % args.print_every_num_batches == 0:
                print(f'iters: {i + 1} | epoch: {epoch + 1} | loss: {loss.item():.4f}')

                speed = (time.time() - time_before) / iter_count
                epochs_left = args.train_epochs - epoch
                time_left = speed * (epochs_left * len_train_loader - i)

                print(f'speed: {speed:.4f} s/iter | time left: {time_left:.4f} s')

                iter_count = 0
                time_before = time.time()

        train_loss_avg = np.mean(train_losses)
        test_loss_avg = eval(args.test_loader, model, criterion, args.len_label)

        time_elapsed = time.time() - epoch_time
        print(
            f'epoch: {epoch + 1} | took: {time_elapsed} s | train loss: {train_loss_avg:.4f} | test loss {test_loss_avg:.4f}'
        )

        if early_stopping(epoch, test_loss_avg):
            print('early stopping')
            break

    # use the best model, not just the last one
    best_model_path = checkpoints_path / 'checkpoint.pt'
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

    return
