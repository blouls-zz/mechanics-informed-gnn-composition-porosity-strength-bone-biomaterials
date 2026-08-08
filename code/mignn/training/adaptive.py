from collections.abc import Iterable

import torch
from torch import nn


def gradient_norm(loss: torch.Tensor, parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    selected = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
    terms = [gradient.square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return loss.new_tensor(0.0)
    return torch.stack(terms).sum().sqrt()


def balanced_weights(data_loss: torch.Tensor, physics_losses: tuple[torch.Tensor, ...], parameters: Iterable[nn.Parameter], previous: torch.Tensor) -> torch.Tensor:
    reusable = list(parameters)
    data_norm = gradient_norm(data_loss, reusable).detach()
    physics_norms = torch.stack([gradient_norm(loss, reusable).detach() for loss in physics_losses])
    ratios = data_norm / physics_norms.clamp_min(1e-12)
    return previous * ratios
