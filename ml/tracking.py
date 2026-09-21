"""
Experiment tracking.  [Addendum 3, section 8]

Lightweight on purpose: a JSON line per run in a directory. The requirement is
that a run be reproducible from its stored configuration, and that is met by
recording enough, not by recording it in a database.

What makes it useful rather than decorative is `reproduce_command`: every
entry can print the exact invocation that would repeat it. A tracker that
stores configuration nobody can replay is a log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .reproducibility import RunRecord


@dataclass
class Experiment:
    """One recorded run."""
    name: str
    run: RunRecord
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    checkpoint: str = ""
    embedding_model: str = ""
    notes: List[str] = field(default_factory=list)
    recorded_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "recorded_utc": self.recorded_utc,
                "run": self.run.to_dict(), "metrics": self.metrics,
                "checkpoint": self.checkpoint, "embedding_model": self.embedding_model,
                "notes": self.notes}

    def reproduce_command(self) -> str:
        """
        The invocation that would repeat this run.

        Refuses when the run is not reproducible rather than printing a command
        that would produce something different: a command that silently does
        not reproduce is worse than no command.
        """
        if not self.run.is_reproducible:
            return (f"# NOT reproducible: {self.run.reproducibility_note()} "
                    f"No command is given, because one that does not reproduce the run "
                    f"is worse than none.")
        config = " ".join(f"--{k} {v}" for k, v in sorted(self.run.config.items()))
        return (f"git checkout {self.run.git_commit}\n"
                f"python -m ml.train --experiment {self.name} "
                f"--seed {self.run.seed} --device {self.run.device} {config}".rstrip())


class Tracker:
    """Append-only JSONL, one line per run."""

    def __init__(self, directory: Path = Path("ml/runs")):
        self.directory = Path(directory)
        self.path = self.directory / "experiments.jsonl"

    def log(self, experiment: Experiment) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(experiment.to_dict()) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def reproducible(self) -> List[Dict[str, Any]]:
        return [e for e in self.all() if e["run"].get("is_reproducible")]

    def summary(self) -> str:
        entries = self.all()
        if not entries:
            return "No runs recorded."
        reproducible = len(self.reproducible())
        return (f"{len(entries)} runs recorded, {reproducible} reproducible. "
                f"{len(entries) - reproducible} were logged from a dirty tree or without "
                f"a resolvable commit and cannot be replayed.")
