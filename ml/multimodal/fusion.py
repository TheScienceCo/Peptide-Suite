"""
Modality-specific encoders and a learned fusion.  [Addendum 3, section 5]

The addendum's requirement is precise and worth restating: modality-specific
encoders plus a fusion layer, *not* concatenated scalars, compared against
single modality and against naive concatenation, with an honest answer about
whether fusion actually helps on held-out data.

The architectures here exist to make that comparison possible, so the naive
baseline is built with the same care as the thing it is meant to lose to. A
fusion layer that beats a concatenation baseline nobody tried to make work is
not evidence of anything.

What "learned fusion" means here: each modality gets its own encoder into a
shared width, and a gate computed from all of them decides how much of each to
use, per example. That is a real fusion mechanism -- the mixture is learned and
input-dependent -- rather than a concatenation with extra layers.

The comparison is run in `ml/experiments/fusion_benefit.py`, which supplies the
verdict rule: overlapping bootstrap intervals mean the experiment did not
distinguish two arms, and that is reported as such rather than as the larger
number winning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence


@dataclass(frozen=True)
class Modality:
    """
    One way of measuring a sequence.

    `describe` is not decoration: a fusion result is uninterpretable without
    knowing what the modalities were, and "modality 0 dominated" says nothing.
    """
    name: str
    featurise: Callable[[str], List[float]]
    describe: str

    def dim(self, example: str) -> int:
        return len(self.featurise(example))


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def single_modality_model(dim: int, width: int = 32):
    """An MLP over one modality. The control each fusion arm has to beat."""
    _, nn = _torch()
    return nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, 1))


def concatenation_model(dims: Sequence[int], width: int = 32):
    """
    Naive concatenation: features joined end to end, one MLP on top.

    Given the same width and the same training budget as the fusion model, so a
    difference between them is about the mechanism rather than about capacity.
    """
    _, nn = _torch()
    return nn.Sequential(nn.Linear(sum(dims), width), nn.ReLU(), nn.Linear(width, 1))


def build_fusion_model(dims: Sequence[int], width: int = 32):
    torch, nn = _torch()

    class GatedFusion(nn.Module):
        """
        Per-modality encoders, then an input-dependent mixture of them.

        The gate is a softmax over modalities computed from the concatenated
        encodings, so the model can lean on one modality for one example and
        another for the next. `last_gate` is kept because a fusion model that
        cannot say which modality it used is a fusion model nobody can check.
        """

        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList(
                nn.Sequential(nn.Linear(d, width), nn.ReLU()) for d in dims
            )
            self.gate = nn.Linear(width * len(dims), len(dims))
            self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            self.last_gate = None

        def forward(self, parts: Sequence["torch.Tensor"]) -> "torch.Tensor":
            encoded = [encoder(part) for encoder, part in zip(self.encoders, parts)]
            weights = torch.softmax(self.gate(torch.cat(encoded, dim=1)), dim=1)
            self.last_gate = weights.detach()
            stacked = torch.stack(encoded, dim=1)              # (batch, modality, width)
            mixed = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            return self.head(mixed)

    return GatedFusion()
