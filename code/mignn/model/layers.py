from typing import Final

import torch
from torch import nn
from torch_geometric.utils import scatter


class EdgeGate(nn.Module):
    def __init__(self, edge_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(edge_dimension, hidden_dimension),
            nn.Softplus(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Sigmoid(),
        )

    def forward(self, edge_attributes: torch.Tensor) -> torch.Tensor:
        return self.layers(edge_attributes)


class EdgeUpdate(nn.Module):
    def __init__(self, hidden_dimension: int, edge_dimension: int) -> None:
        super().__init__()
        width: Final = 2 * hidden_dimension + edge_dimension
        self.layers = nn.Sequential(
            nn.Linear(width, hidden_dimension),
            nn.Softplus(),
            nn.Linear(hidden_dimension, edge_dimension),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return edge + self.layers(torch.cat((source, target, edge), dim=-1))


class EdgeGatedConvolution(nn.Module):
    def __init__(self, hidden_dimension: int, edge_dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.gate = EdgeGate(edge_dimension, hidden_dimension)
        self.edge_update = EdgeUpdate(hidden_dimension, edge_dimension)
        self.activation = nn.Softplus()

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sources = edge_index[0]
        targets = edge_index[1]
        messages = self.gate(edges) * self.activation(self.projection(nodes[sources]))
        aggregate = scatter(messages, targets, dim=0, dim_size=nodes.shape[0], reduce="sum")
        next_nodes = nodes + aggregate
        next_edges = self.edge_update(next_nodes[sources], next_nodes[targets], edges)
        return next_nodes, next_edges


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dimension, 1, bias=False)

    def forward(self, nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        logits = self.attention(nodes).squeeze(-1)
        maximum = scatter(logits, batch, dim=0, reduce="max")[batch]
        weights = torch.exp(logits - maximum)
        denominator = scatter(weights, batch, dim=0, reduce="sum")[batch]
        normalized = weights / denominator.clamp_min(1e-12)
        return scatter(nodes * normalized.unsqueeze(-1), batch, dim=0, reduce="sum")
