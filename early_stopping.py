from pathlib import Path
from tabnanny import verbose
from typing import Optional, TypedDict

import torch
from torch import nn
from torch.optim import Optimizer


class Checkpoint(TypedDict):
    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    val_loss: float


class EarlyStopping:
    best_score: Optional[float] = None
    val_loss_min = float('inf')
    counter = 0
    early_stop = False

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        checkpoints_dir: str,
        patience: int,
        delta: float = 0.0,
        verbose: bool = False,
    ):
        """Early stopping in PyTorch.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            checkpoints_dir: Checkpoints directory.
            patience: How many consecutive calls with a non-improving loss to wait before early stopping. Must be > 0.
            delta: How much the loss must decrease to be considered as an improvement. Must be > 0.0.
            verbose: If True, prints a message for each time validation loss doesn't improve and when a checkpoint is saved;
                otherwise, prints nothing.
        """
        assert patience > 0
        assert delta >= 0.0

        self.model = model
        self.optimizer = optimizer
        self.checkpoints_dir = checkpoints_dir
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

    def __call__(
        self,
        epoch: int,
        val_loss: float,
    ) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if verbose:
                print(f'early stopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss)
        return False

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
    ):
        checkpoints_path = Path(self.checkpoints_dir)
        if self.verbose:
            print(
                f'validation loss changed ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...'
            )
        torch.save(
            Checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
            ),
            checkpoints_path / 'checkpoint.pt',
        )
        self.val_loss_min = val_loss
