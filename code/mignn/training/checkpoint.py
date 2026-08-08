import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from mignn.training.state import TrainingState


def save_parameters(path: Path, model: nn.Module, optimizer: Optimizer, state: TrainingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state.__dict__, "rng": torch.get_rng_state()}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_parameters(path: Path, model: nn.Module, optimizer: Optimizer) -> TrainingState:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"])
    return TrainingState(**payload["state"])
