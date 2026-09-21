"""
SAPT contact decomposition, and the sum that is forbidden.
[Addendum 2 section 5, post-6i]

SAPT0 or DFT-SAPT on truncated contact pairs, decomposed into electrostatics,
exchange-repulsion, induction and dispersion.

The purpose is explanation and substitution reasoning, not affinity prediction,
and the components are useful precisely because they respond to different design
moves:

  A dispersion-dominated contact responds to volume and polarizability.
  An electrostatics-dominated contact responds to charge, geometry, desolvation.
  An induction-heavy contact flags where a fixed-charge force field will
  misbehave and a polarizable model or a QM region is warranted.

So every proposal that modifies a contact must name which component it targets.
"Improves the interaction" is not a mechanism; "increases dispersion by
enlarging the buried hydrophobic surface" is one, and it is falsifiable.

FORBIDDEN: summing the components into a predicted binding free energy.
Interaction energy is not binding free energy. It omits desolvation and entropy
entirely, and the omission is not small -- it is frequently larger than the
whole interaction. Affinity estimates come from FEP/TI or from a trained scorer,
never from this module. The prohibition is enforced in code below rather than
documented, because the sum is one line to write and looks like the obvious
thing to do with four numbers that share a unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SAPTComponent(Enum):
    """The four terms. Each responds to a different kind of design move."""
    ELECTROSTATICS = "electrostatics"
    EXCHANGE_REPULSION = "exchange_repulsion"
    INDUCTION = "induction"
    DISPERSION = "dispersion"

    @property
    def responds_to(self) -> str:
        return {
            SAPTComponent.ELECTROSTATICS:
                "charge, geometry and desolvation",
            SAPTComponent.EXCHANGE_REPULSION:
                "steric overlap — it is the penalty for putting atoms where other atoms "
                "already are, so it responds to size and to fit",
            SAPTComponent.INDUCTION:
                "polarizability and field strength. A contact dominated by this term is a "
                "warning as much as a handle: a fixed-charge force field will misbehave "
                "here, and a polarizable model or a QM region is warranted",
            SAPTComponent.DISPERSION:
                "volume and polarizability — buried hydrophobic surface, aromatic contact "
                "area, halogen substitution",
        }[self]


class AffinitySummationError(Exception):
    """
    Raised when SAPT components are summed into a predicted binding free energy.

    Named for what it prevents rather than for what went wrong, because the
    error is not a mistake in arithmetic. The sum is arithmetically fine; it is
    the interpretation that is wrong, and the interpretation is the whole reason
    someone reaches for the sum.
    """


@dataclass(frozen=True)
class ContactDecomposition:
    """
    One interface contact, decomposed.

    Deliberately offers no total. The components are exposed individually and
    `interaction_energy` exists so that the one legitimate sum -- the total
    interaction energy, which is a real quantity -- can be taken while being
    labelled as what it is and not as an affinity.
    """
    residue_a: str
    residue_b: str
    electrostatics: float
    exchange_repulsion: float
    induction: float
    dispersion: float
    method: str
    basis_set: str
    truncation_scheme: str

    @property
    def components(self) -> Dict[SAPTComponent, float]:
        return {
            SAPTComponent.ELECTROSTATICS: self.electrostatics,
            SAPTComponent.EXCHANGE_REPULSION: self.exchange_repulsion,
            SAPTComponent.INDUCTION: self.induction,
            SAPTComponent.DISPERSION: self.dispersion,
        }

    @property
    def dominant(self) -> SAPTComponent:
        """
        The component contributing most. Compared by magnitude, since
        exchange-repulsion is positive and the attractive terms are negative,
        and a signed comparison would always return the repulsion.
        """
        return max(self.components, key=lambda c: abs(self.components[c]))

    def interaction_energy(self) -> float:
        """
        The total interaction energy. A real quantity, and NOT an affinity.

        Kept available because it is legitimately the sum of the components, and
        named so that a caller writing `interaction_energy()` has said what they
        are taking. `predicted_affinity` does not exist, and calling
        `as_binding_free_energy` raises.
        """
        return (self.electrostatics + self.exchange_repulsion
                + self.induction + self.dispersion)

    def as_binding_free_energy(self):
        """The forbidden operation, present so that it fails loudly."""
        raise AffinitySummationError(
            f"SAPT components for {self.residue_a}-{self.residue_b} may not be summed "
            f"into a binding free energy. Interaction energy is not binding free energy: "
            f"it omits desolvation and entropy entirely, and that omission is frequently "
            f"larger than the interaction itself. Affinity estimates come from FEP/TI or "
            f"from a trained scorer, never from this decomposition. "
            f"Use interaction_energy() if the interaction energy is what is wanted."
        )

    def summary(self) -> str:
        dominant = self.dominant
        return (f"{self.residue_a}-{self.residue_b}: {dominant.value}-dominated "
                f"({self.components[dominant]:+.2f}). Responds to {dominant.responds_to}.")


@dataclass
class ComponentClaim:
    """
    A proposal's statement about which component it targets.

    Required of every proposal that modifies a contact. Without it "improves the
    interaction" stands as a mechanism, and it is not one -- it is the
    conclusion restated.
    """
    proposal: str
    targeted: Optional[SAPTComponent]
    justification: str = ""

    @property
    def is_stated(self) -> bool:
        return self.targeted is not None

    def violation(self) -> str:
        if self.is_stated:
            return ""
        return (
            f"'{self.proposal}' modifies a contact without naming which SAPT component it "
            f"targets. A dispersion-dominated contact responds to volume and "
            f"polarizability; an electrostatics-dominated one responds to charge and "
            f"desolvation. Without the component the proposal has given its conclusion in "
            f"place of its mechanism."
        )


# Language that indicates which component a proposal is reaching for. Used to
# read a claim out of a proposal's own words, never to invent one: an
# unrecognised proposal returns no component and is reported as not having
# named it.
_COMPONENT_LANGUAGE = {
    SAPTComponent.DISPERSION: (
        "hydrophobic", "buried surface", "aromatic", "volume", "polarizab",
        "van der waals", "packing", "halogen"),
    SAPTComponent.ELECTROSTATICS: (
        "charge", "salt bridge", "ionic", "electrostatic", "desolvat", "dipole"),
    SAPTComponent.INDUCTION: (
        "induction", "polarisation", "polarization", "induced dipole"),
    SAPTComponent.EXCHANGE_REPULSION: (
        "steric clash", "steric overlap", "too large", "relieve strain", "exchange"),
}


def read_component_claim(proposal: str) -> ComponentClaim:
    """Which component a proposal names, if any, from its own words."""
    text = (proposal or "").lower()
    for component, markers in _COMPONENT_LANGUAGE.items():
        if any(marker in text for marker in markers):
            return ComponentClaim(proposal=proposal, targeted=component,
                                  justification=f"names {component.value} language")
    return ComponentClaim(proposal=proposal, targeted=None)


class DecompositionNotAvailable(NotImplementedError):
    """Raised when a decomposition is requested and cannot be produced."""


def decompose(residue_a: str, residue_b: str, **_kwargs) -> ContactDecomposition:
    """
    Run SAPT on a contact pair. Not implemented; raises.

    Returning plausible components would be worse here than almost anywhere
    else in this system, because the components are used to choose a design
    direction. A fabricated dispersion-dominated result sends someone to enlarge
    a hydrophobic surface on a contact that is actually electrostatic.
    """
    raise DecompositionNotAvailable(
        f"No SAPT decomposition was produced for {residue_a}-{residue_b}. It needs a QM "
        f"package with SAPT0 or DFT-SAPT, an interface structure to truncate contact "
        f"pairs from, and a declared truncation scheme. None is available in this "
        f"deployment, so contacts are not decomposed and no component is attributed."
    )
