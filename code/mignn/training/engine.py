import logging
from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Batch

from mignn.physics.objective import MechanicsObjective, PhysicsBatch
from mignn.training.state import TrainingState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerSettings:
    pretrain_epochs: int
    joint_epochs: int
    learning_rate: float
    weight_decay: float
    restart_period: int
    patience: int


class Trainer:
    def __init__(self, model: nn.Module, objective: MechanicsObjective, settings: TrainerSettings) -> None:
        self.model = model
        self.objective = objective
        self.settings = settings
        self.optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=settings.restart_period)

    def train_batch(self, graph: Batch, physics: PhysicsBatch, data_enabled: bool) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(graph)
        losses = self.objective(prediction, physics, data_enabled)
        losses.total.backward()
        self.optimizer.step()
        return float(losses.total.detach())

    def run_epoch(self, batches: Iterable[tuple[Batch, PhysicsBatch]], data_enabled: bool) -> float:
        total = 0.0
        count = 0
        for graph, physics in batches:
            total += self.train_batch(graph, physics, data_enabled)
            count += 1
        self.scheduler.step()
        return total / max(count, 1)

    def fit(self, batches: Iterable[tuple[Batch, PhysicsBatch]], state: TrainingState) -> TrainingState:
        cached = list(batches)
        for epoch in range(self.settings.pretrain_epochs):
            loss = self.run_epoch(cached, False)
            state.epoch = epoch + 1
            state.phase = "physics"
            logger.info("phase=%s epoch=%d loss=%.8f", state.phase, state.epoch, loss)
        for epoch in range(self.settings.joint_epochs):
            loss = self.run_epoch(cached, True)
            state.epoch = self.settings.pretrain_epochs + epoch + 1
            state.phase = "joint"
            logger.info("phase=%s epoch=%d loss=%.8f", state.phase, state.epoch, loss)
        return state
