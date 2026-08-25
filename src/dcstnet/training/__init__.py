"""Two-phase training and artifact materialization."""

from .phase_a import materialize_trends, train_phase_a
from .phase_b import train_phase_b

__all__ = ["materialize_trends", "train_phase_a", "train_phase_b"]
