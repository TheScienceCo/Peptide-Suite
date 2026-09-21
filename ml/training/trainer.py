"""
Training loop: checkpoints, early stopping, and what gets recorded.
[Addendum 3, sections 1 and 8]

The loop itself is unremarkable and should be. What is deliberate is what it
refuses to lose: the run record travels with the checkpoint, so a saved model
can always say which commit, seed, device and dataset version produced it. A
checkpoint without that is a file of weights nobody can reproduce or withdraw.

Early stopping restores the best weights rather than keeping the last ones.
Keeping the last is the common default and it is wrong: the run stopped because
the metric got worse, so the final weights are by construction not the ones that
scored best.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..reproducibility import RunRecord


@dataclass
class TrainingConfig:
    """Everything that changes a result, in one object that can be serialised."""
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    mode: str = "min"
    seed: int = 0
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """What a run produced, and whether it can be reproduced."""
    run: RunRecord
    config: TrainingConfig
    history: List[EpochRecord] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_value: Optional[float] = None
    stopped_early: bool = False
    checkpoint_path: Optional[str] = None

    @property
    def epochs_run(self) -> int:
        return len(self.history)

    def summary(self) -> str:
        stop = (f"stopped early at epoch {self.epochs_run}" if self.stopped_early
                else f"ran all {self.epochs_run} epochs")
        best = (f"best {self.config.monitor} {self.best_value:.4f} at epoch "
                f"{self.best_epoch}" if self.best_value is not None else "no best recorded")
        return f"{stop}; {best}. {self.run.reproducibility_note()}"


class EarlyStopping:
    """
    Patience on a monitored metric, with the best weights kept.

    `min_delta` guards against counting numerical noise as improvement. Without
    it a run can continue indefinitely on changes in the sixth decimal place,
    which is not early stopping, it is a slower way to overfit.
    """

    def __init__(self, patience: int, min_delta: float, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.waited = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Returns True when training should stop."""
        if self.is_improvement(value):
            self.best, self.best_epoch, self.waited = value, epoch, 0
            return False
        self.waited += 1
        return self.waited >= self.patience


def record_path(checkpoint: Path) -> Path:
    """Where a checkpoint's run record lives. Appended, never substituted."""
    return Path(str(checkpoint) + ".run.json")


def save_checkpoint(path: Path, model, result: TrainingResult) -> None:
    """
    Save weights next to the run record that produced them.

    Written as two files rather than one pickle: the record stays readable
    without torch, which is what makes a model card generatable from a
    checkpoint directory by something other than this codebase.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    # Appended rather than with_suffix: model.pt and model.bin both map to
    # model.run.json under with_suffix, so two checkpoints in one directory
    # would share -- and silently overwrite -- a single record.
    record_path(path).write_text(json.dumps({
        "run": result.run.to_dict(),
        "config": result.config.to_dict(),
        "best_epoch": result.best_epoch,
        "best_value": result.best_value,
        "stopped_early": result.stopped_early,
        "epochs_run": result.epochs_run,
    }, indent=2) + "\n")


def load_checkpoint(path: Path, model):
    """
    Load weights and the record. Raises when the record is missing.

    A checkpoint without its record cannot say which commit, seed or dataset
    produced it, so it cannot be reproduced, compared, or withdrawn. Loading it
    anyway would put an unattributable model into service.
    """
    import torch

    path = Path(path)
    record = record_path(path)
    if not record.exists():
        raise FileNotFoundError(
            f"{path.name} has no run record at {record.name}. A checkpoint that "
            f"cannot say which commit, seed and dataset produced it cannot be "
            f"reproduced, compared against a later version, or withdrawn when that "
            f"version is found to be wrong.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model, json.loads(record.read_text())


def train(model, train_batches: Callable, val_batches: Optional[Callable],
          loss_fn, config: TrainingConfig, name: str = "run",
          dataset_version: str = "", checkpoint: Optional[Path] = None) -> TrainingResult:
    """
    Fit a model, stopping on patience and restoring the best weights.

    `train_batches` and `val_batches` are callables returning fresh iterables,
    not iterables: an iterator passed here would be exhausted after one epoch
    and every later epoch would train on nothing while reporting a loss of
    zero.
    """
    import copy
    import torch

    run = RunRecord.capture(name, seed=config.seed, device=config.device,
                            dataset_version=dataset_version, config=config.to_dict())
    device = torch.device(run.device)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    stopper = EarlyStopping(config.patience, config.min_delta, config.mode)
    result = TrainingResult(run=run, config=config)
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_batches():
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(x)
            seen += len(x)
        train_loss = total / max(seen, 1)

        val_loss = None
        if val_batches is not None:
            model.eval()
            total, seen = 0.0, 0
            with torch.no_grad():
                for x, y in val_batches():
                    x, y = x.to(device), y.to(device)
                    total += float(loss_fn(model(x), y)) * len(x)
                    seen += len(x)
            val_loss = total / max(seen, 1)

        result.history.append(EpochRecord(epoch=epoch, train_loss=train_loss,
                                          val_loss=val_loss))

        monitored = val_loss if val_loss is not None else train_loss
        if math.isnan(monitored):
            result.stopped_early = True
            result.history[-1].metrics["stopped_on"] = float("nan")
            break
        if stopper.is_improvement(monitored):
            best_weights = copy.deepcopy(model.state_dict())
        if stopper.step(monitored, epoch):
            result.stopped_early = True
            break

    # The run stopped because the metric got worse, so the final weights are by
    # construction not the best ones.
    model.load_state_dict(best_weights)
    result.best_epoch, result.best_value = stopper.best_epoch, stopper.best

    if checkpoint is not None:
        save_checkpoint(Path(checkpoint), model, result)
        result.checkpoint_path = str(checkpoint)
    return result
