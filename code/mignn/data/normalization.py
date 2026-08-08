from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Standardization:
    mean: torch.Tensor
    standard_deviation: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.standard_deviation

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.standard_deviation + self.mean


def fit_standardization(values: torch.Tensor) -> Standardization:
    mean = values.mean(dim=0)
    standard_deviation = values.std(dim=0, unbiased=False).clamp_min(1e-8)
    return Standardization(mean, standard_deviation)
