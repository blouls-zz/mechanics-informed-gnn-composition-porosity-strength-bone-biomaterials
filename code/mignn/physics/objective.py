from dataclasses import dataclass

import torch
from torch import nn

from mignn.types import PropertyPrediction


@dataclass(frozen=True)
class PhysicsBatch:
    targets: torch.Tensor
    solid_strength: torch.Tensor
    solid_density: torch.Tensor
    density: torch.Tensor
    elastic_limit_strain: torch.Tensor
    target_variance: torch.Tensor


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    data: torch.Tensor
    gibson_ashby: torch.Tensor
    hooke: torch.Tensor
    yield_envelope: torch.Tensor


class MechanicsObjective(nn.Module):
    def __init__(self, gibson_ashby: float, hooke: float, yield_envelope: float) -> None:
        super().__init__()
        self.gibson_ashby_weight = gibson_ashby
        self.hooke_weight = hooke
        self.yield_weight = yield_envelope

    def forward(self, prediction: PropertyPrediction, batch: PhysicsBatch, data_enabled: bool = True) -> LossTerms:
        stacked = torch.stack(
            (prediction.compressive_strength, prediction.elastic_modulus, prediction.yield_strength),
            dim=-1,
        )
        task_weights = batch.target_variance.reciprocal()
        data = ((stacked - batch.targets).square() * task_weights).mean()
        if not data_enabled:
            data = data * 0.0
        relative_density = batch.density / batch.solid_density
        scaled_strength = prediction.coefficient * relative_density.pow(prediction.exponent)
        gibson = (prediction.compressive_strength / batch.solid_strength - scaled_strength).square().mean()
        strain_mask = (batch.elastic_limit_strain < 0.02).to(stacked.dtype)
        hooke_residual = prediction.elastic_modulus - prediction.compressive_strength / batch.elastic_limit_strain
        hooke = (strain_mask * hooke_residual.square()).mean()
        ceiling = batch.solid_strength * relative_density
        yield_term = torch.relu(prediction.compressive_strength - ceiling).square().mean()
        total = data + self.gibson_ashby_weight * gibson + self.hooke_weight * hooke + self.yield_weight * yield_term
        return LossTerms(total, data, gibson, hooke, yield_term)
