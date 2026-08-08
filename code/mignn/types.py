from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch


class PropertyPrediction(NamedTuple):
    compressive_strength: torch.Tensor
    elastic_modulus: torch.Tensor
    yield_strength: torch.Tensor
    relative_density: torch.Tensor
    coefficient: torch.Tensor
    exponent: torch.Tensor


@dataclass(frozen=True)
class FeatureDimensions:
    node: int
    edge: int
    global_state: int
    hidden: int


@dataclass(frozen=True)
class SplitFractions:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class RunPaths:
    root: Path
    logs: Path
    parameters: Path
    predictions: Path
