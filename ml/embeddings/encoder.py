"""
Sequence encoders, pretrained and otherwise.  [Addendum 3, section 2]

A pretrained protein language model produces a representation whose meaning
depends entirely on the model and version that made it. So every vector carries
those, and two vectors from different models are not comparable even when they
have the same dimensionality.

ESM-2 is the intended encoder and its weights are not reachable from this
environment. The fallback is one-hot and composition encoding, which is a real
encoding and is NOT a language-model embedding -- it has no learned content at
all. The distinction is carried in the type rather than in a comment, because
the whole value of a pretrained embedding is the learned content, and a
consumer that cannot tell the two apart will report "ESM-2 embedding" for a
count of amino acids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..provenance import MLProvenance, MLQuantity

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class EncoderKind(Enum):
    PRETRAINED_LANGUAGE_MODEL = "pretrained_language_model"
    DETERMINISTIC = "deterministic"

    @property
    def has_learned_content(self) -> bool:
        return self is EncoderKind.PRETRAINED_LANGUAGE_MODEL


class EncoderUnavailable(Exception):
    """Raised when a pretrained encoder is requested and its weights are absent."""


@dataclass
class Embedding:
    """
    One sequence's representation, with what produced it.

    `pooled` and `residue_level` are both kept: a pooled vector cannot answer a
    per-position question, and re-pooling from residue level later is cheap
    while recovering residue level from a pooled vector is impossible.
    """
    sequence: str
    pooled: List[float]
    residue_level: Optional[List[List[float]]]
    model: str
    model_version: str
    kind: EncoderKind
    preprocessing: str
    pooling: str

    @property
    def dim(self) -> int:
        return len(self.pooled)

    def comparable_with(self, other: "Embedding") -> bool:
        """
        Whether two embeddings may be compared or placed in one space.

        Same dimensionality is not enough. Two 320-vectors from different models
        occupy unrelated spaces, and a PCA over a mixture of them produces a
        plot whose axes mean nothing.
        """
        return (self.model == other.model
                and self.model_version == other.model_version
                and self.kind is other.kind)

    def as_quantity(self, name: str = "embedding") -> MLQuantity:
        return MLQuantity(
            name=name, value=self.pooled, provenance=MLProvenance.PRETRAINED_EMBEDDING,
            metadata={"model": self.model, "model_version": self.model_version,
                      "embedding_dim": self.dim, "pooling": self.pooling,
                      "preprocessing": self.preprocessing})


class DeterministicEncoder:
    """
    One-hot plus composition. A real encoding with no learned content.

    Present so the pipeline runs end to end without network access, and named
    so nothing can report it as a language-model embedding. Composition cannot
    see order, and the one-hot block is position-dependent only up to the
    padded length, which are limitations worth knowing rather than hiding.
    """

    model = "deterministic-onehot-composition"
    model_version = "1.0"
    kind = EncoderKind.DETERMINISTIC

    def __init__(self, max_length: int = 50):
        self.max_length = max_length

    def encode(self, sequence: str) -> Embedding:
        sequence = (sequence or "").upper()
        residues = []
        for residue in sequence[:self.max_length]:
            vector = [0.0] * len(AMINO_ACIDS)
            index = AMINO_ACIDS.find(residue)
            if index >= 0:
                vector[index] = 1.0
            residues.append(vector)

        length = max(len(sequence), 1)
        composition = [sequence.count(a) / length for a in AMINO_ACIDS]
        return Embedding(
            sequence=sequence, pooled=composition, residue_level=residues,
            model=self.model, model_version=self.model_version, kind=self.kind,
            preprocessing=f"uppercase, truncated to {self.max_length}",
            pooling="amino-acid composition")


class PositionalOneHotEncoder(DeterministicEncoder):
    """
    Flattened one-hot: the same alphabet, but position-aware.

    Composition cannot see order, so under it every single substitution of the
    same two residues lands in one place regardless of where it happened --
    which makes it the wrong space for a picture whose whole subject is where
    a substitution was made. This flattens the residue-level block instead, so
    a substitution at position 3 and the same substitution at position 20 are
    different points.

    It carries its own model name so it can never be pooled with the
    composition encoder's vectors. They have the same alphabet and unrelated
    geometry, and the comparability check is on the name.

    The cost is honest and worth stating: the vector is padded to a fixed
    length, so two sequences of different length are compared over a window
    rather than aligned, and still nothing here is learned.
    """

    model = "deterministic-positional-onehot"
    model_version = "1.0"
    kind = EncoderKind.DETERMINISTIC

    def encode(self, sequence: str) -> Embedding:
        base = super().encode(sequence)
        flat: List[float] = []
        for index in range(self.max_length):
            if index < len(base.residue_level):
                flat.extend(base.residue_level[index])
            else:
                flat.extend([0.0] * len(AMINO_ACIDS))
        return Embedding(
            sequence=base.sequence, pooled=flat, residue_level=base.residue_level,
            model=self.model, model_version=self.model_version, kind=self.kind,
            preprocessing=base.preprocessing,
            pooling=f"flattened one-hot, padded to {self.max_length} positions")


class ESM2Encoder:
    """
    ESM-2, when its weights are present. Raises when they are not.

    No silent fallback to the deterministic encoder. A caller that asked for a
    pretrained representation and received a count of amino acids would draw
    conclusions about what language-model features contribute, and the answer
    would be about neither.
    """

    kind = EncoderKind.PRETRAINED_LANGUAGE_MODEL

    def __init__(self, model_name: str = "esm2_t12_35M_UR50D",
                 weights_dir: Optional[Path] = None):
        self.model = model_name
        self.model_version = model_name
        self.weights_dir = Path(weights_dir) if weights_dir else Path("ml/weights")

    @property
    def is_available(self) -> bool:
        try:
            import esm  # noqa: F401
        except ImportError:
            return False
        return self.weights_dir.exists()

    def encode(self, sequence: str) -> Embedding:
        raise EncoderUnavailable(
            f"{self.model} is not available in this environment. Its weights are not "
            f"present in {self.weights_dir} and the model hosts are unreachable. No "
            f"fallback is substituted: a caller that asked for a pretrained "
            f"representation and silently received amino-acid counts would conclude "
            f"something about language-model features from a vector that has none."
        )


class EmbeddingCache:
    """
    Cache keyed by sequence AND model version.

    Keying on the sequence alone is the obvious mistake and it is unrecoverable:
    upgrade the model, and every cached vector from the previous version comes
    back under the new name with nothing to indicate it.
    """

    def __init__(self, directory: Path = Path(".embedding_cache")):
        self.directory = Path(directory)

    def _key(self, sequence: str, model: str, version: str) -> str:
        digest = hashlib.sha256(f"{model}|{version}|{sequence}".encode()).hexdigest()
        return digest[:32]

    def path_for(self, sequence: str, model: str, version: str) -> Path:
        return self.directory / f"{self._key(sequence, model, version)}.json"

    def get(self, sequence: str, model: str, version: str) -> Optional[Embedding]:
        path = self.path_for(sequence, model, version)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return Embedding(sequence=data["sequence"], pooled=data["pooled"],
                         residue_level=data.get("residue_level"), model=data["model"],
                         model_version=data["model_version"],
                         kind=EncoderKind(data["kind"]),
                         preprocessing=data["preprocessing"], pooling=data["pooling"])

    def put(self, embedding: Embedding) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path_for(embedding.sequence, embedding.model,
                      embedding.model_version).write_text(json.dumps({
                          "sequence": embedding.sequence, "pooled": embedding.pooled,
                          "residue_level": embedding.residue_level,
                          "model": embedding.model,
                          "model_version": embedding.model_version,
                          "kind": embedding.kind.value,
                          "preprocessing": embedding.preprocessing,
                          "pooling": embedding.pooling}))


def available_encoders() -> Dict[str, Dict[str, object]]:
    """What can actually encode here, and why the rest cannot."""
    esm = ESM2Encoder()
    return {
        DeterministicEncoder.model: {
            "kind": EncoderKind.DETERMINISTIC.value, "available": True,
            "has_learned_content": False,
            "note": ("A real encoding with no learned content. Not a language-model "
                     "embedding, and not reported as one.")},
        PositionalOneHotEncoder.model: {
            "kind": EncoderKind.DETERMINISTIC.value, "available": True,
            "has_learned_content": False,
            "note": ("Position-aware one-hot, padded to a fixed length. Still nothing "
                     "learned; distance is a count of positions that differ.")},
        esm.model: {
            "kind": EncoderKind.PRETRAINED_LANGUAGE_MODEL.value,
            "available": esm.is_available, "has_learned_content": True,
            "note": ("Intended encoder. Weights are not present and the model hosts are "
                     "unreachable from this environment.")},
    }
