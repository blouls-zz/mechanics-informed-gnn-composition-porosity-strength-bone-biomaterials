from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    mean_absolute_error: float
    root_mean_squared_error: float


def regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> RegressionMetrics:
    residual = target - prediction
    centered = target - target.mean()
    r2 = 1.0 - residual.square().sum() / centered.square().sum().clamp_min(1e-12)
    mae = residual.abs().mean()
    rmse = residual.square().mean().sqrt()
    return RegressionMetrics(float(r2), float(mae), float(rmse))


@dataclass(frozen=True)
class ViolationRates:
    gibson_ashby: float
    hooke: float
    yield_envelope: float


def violation_rates(compressive: torch.Tensor, modulus: torch.Tensor, ga_strength: torch.Tensor, strain: torch.Tensor, ceiling: torch.Tensor) -> ViolationRates:
    ga = ((compressive < 0.5 * ga_strength) | (compressive > 2.0 * ga_strength)).float().mean()
    hooke = ((modulus - compressive / strain).abs() > 0.5 * modulus).float().mean()
    yield_rate = ((compressive > ceiling) | (compressive < 0.0)).float().mean()
    return ViolationRates(float(ga), float(hooke), float(yield_rate))
