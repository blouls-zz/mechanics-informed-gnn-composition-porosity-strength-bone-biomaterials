import torch
from torch import nn
from torch_geometric.data import Batch

from mignn.model.layers import AttentionReadout, EdgeGatedConvolution
from mignn.types import FeatureDimensions, PropertyPrediction


class PositiveHead(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Softplus(),
            nn.Linear(hidden_dimension, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class MechanicsInformedGraphNetwork(nn.Module):
    def __init__(self, dimensions: FeatureDimensions, layers: int = 4) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(dimensions.node, dimensions.hidden)
        self.edge_encoder = nn.Linear(dimensions.edge, dimensions.hidden)
        self.convolutions = nn.ModuleList(
            EdgeGatedConvolution(dimensions.hidden, dimensions.hidden) for _ in range(layers)
        )
        self.readout = AttentionReadout(dimensions.hidden)
        graph_width = dimensions.hidden + dimensions.global_state
        self.backbone = nn.Sequential(
            nn.Linear(graph_width, dimensions.hidden),
            nn.Softplus(),
            nn.Linear(dimensions.hidden, dimensions.hidden),
            nn.Softplus(),
        )
        self.relative_density = nn.Linear(dimensions.hidden, 1)
        self.coefficient = nn.Linear(dimensions.hidden, 1)
        self.exponent = nn.Linear(dimensions.hidden, 1)
        self.compressive_strength = PositiveHead(dimensions.hidden, dimensions.hidden)
        self.elastic_modulus = PositiveHead(dimensions.hidden, dimensions.hidden)
        self.yield_strength = PositiveHead(dimensions.hidden, dimensions.hidden)

    def forward(self, graph: Batch) -> PropertyPrediction:
        nodes = self.node_encoder(graph.x)
        edges = self.edge_encoder(graph.edge_attr)
        for convolution in self.convolutions:
            nodes, edges = convolution(nodes, graph.edge_index, edges)
        pooled = self.readout(nodes, graph.batch)
        state = torch.cat((pooled, graph.u), dim=-1)
        representation = self.backbone(state)
        relative_density = torch.sigmoid(self.relative_density(representation)).squeeze(-1)
        coefficient = 0.1 + 0.9 * torch.sigmoid(self.coefficient(representation)).squeeze(-1)
        exponent = 1.0 + 2.0 * torch.sigmoid(self.exponent(representation)).squeeze(-1)
        return PropertyPrediction(
            self.compressive_strength(representation),
            self.elastic_modulus(representation),
            self.yield_strength(representation),
            relative_density,
            coefficient,
            exponent,
        )
