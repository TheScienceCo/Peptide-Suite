"""
Seeds, devices, and the run record.  [Addendum 3, section 1 and 8]

A result that cannot be reproduced is an anecdote. Everything here exists so
that a stored configuration is enough to get the same numbers back: the seed,
the device, the library versions, the git commit, and the dataset version.

Device selection includes MPS because development happens on an M-series Mac,
and MPS is not merely "CUDA but slower" -- it has its own numerical behaviour,
so the device is part of the run record rather than an implementation detail.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def select_device(preference: str = "auto") -> str:
    """
    The device to run on. "auto" prefers CUDA, then MPS, then CPU.

    Returned as a string rather than a torch.device so that a run record can
    hold it without importing torch to read it back.
    """
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Seed every source of randomness, and report what could not be made
    deterministic.

    The report matters. Some kernels have no deterministic implementation, and
    a run that silently used one is not reproducible even though the seed was
    set. Saying so is the difference between a reproducible result and one that
    merely looks like it.
    """
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        numpy_seeded = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    caveats = []
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            caveats.append(f"deterministic algorithms unavailable: {e}")
        if select_device() == "mps":
            caveats.append(
                "MPS does not guarantee bitwise reproducibility across releases; the "
                "device is recorded so a mismatch can be traced to it rather than to "
                "the code.")

    return {"seed": seed, "numpy_seeded": numpy_seeded,
            "deterministic_requested": deterministic, "caveats": caveats}


def git_commit() -> str:
    """The commit a run was made from, or a marker saying it is unknown."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                   text=True, timeout=5).stdout.strip()
            return f"{sha}-dirty" if dirty else sha
    except Exception:
        pass
    return "UNKNOWN"


@dataclass
class RunRecord:
    """
    Everything needed to reproduce a run.

    A dirty git commit is recorded as dirty rather than as the commit it is
    closest to. "Reproducible from abc123" is false when the working tree had
    uncommitted changes, and the suffix is the only place that fact survives.
    """
    name: str
    seed: int
    device: str
    git_commit: str
    dataset_version: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    library_versions: Dict[str, str] = field(default_factory=dict)
    seeding_caveats: list = field(default_factory=list)

    @classmethod
    def capture(cls, name: str, seed: int, device: str = "auto",
                dataset_version: str = "", config: Optional[Dict] = None) -> "RunRecord":
        import torch
        resolved = select_device(device)
        seeding = seed_everything(seed)
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform()}
        try:
            import numpy as np
            versions["numpy"] = np.__version__
        except ImportError:
            pass
        return cls(name=name, seed=seed, device=resolved, git_commit=git_commit(),
                   dataset_version=dataset_version, config=dict(config or {}),
                   library_versions=versions, seeding_caveats=seeding["caveats"])

    @property
    def is_reproducible(self) -> bool:
        """
        Whether this run can be reproduced from its record.

        A dirty tree means no: the commit does not describe the code that ran.
        """
        return not self.git_commit.endswith("-dirty") and self.git_commit != "UNKNOWN"

    def reproducibility_note(self) -> str:
        if self.is_reproducible:
            note = f"Reproducible from commit {self.git_commit[:12]} with seed {self.seed}."
        elif self.git_commit.endswith("-dirty"):
            note = ("NOT reproducible: the working tree had uncommitted changes, so the "
                    "recorded commit does not describe the code that ran.")
        else:
            note = "NOT reproducible: the git commit could not be determined."
        if self.seeding_caveats:
            note += " " + " ".join(self.seeding_caveats)
        return note

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "device": self.device,
                "git_commit": self.git_commit, "dataset_version": self.dataset_version,
                "config": self.config, "library_versions": self.library_versions,
                "seeding_caveats": self.seeding_caveats,
                "is_reproducible": self.is_reproducible}
