from dataclasses import dataclass

import torch


@dataclass
class TrainingState:
    epoch: int
    phase: str
    best_r2: float
    epochs_without_improvement: int
    seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
