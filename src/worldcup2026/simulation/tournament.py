"""48-team tournament simulator: group stage + expanded knockout bracket.

See METHODOLOGY.md §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    n_runs: int = 20_000
    seed: int = 42


def simulate_tournament(*args, **kwargs):
    """Run N Monte Carlo simulations and return per-team stage probabilities. v1."""
    raise NotImplementedError


def simulate_group_stage(*args, **kwargs):
    """Simulate a group stage and apply FIFA tiebreakers. v1."""
    raise NotImplementedError


def simulate_knockout(*args, **kwargs):
    """Simulate the knockout bracket including ET and penalties. v1."""
    raise NotImplementedError
