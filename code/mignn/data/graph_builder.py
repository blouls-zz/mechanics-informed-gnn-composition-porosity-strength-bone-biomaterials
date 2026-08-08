from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from mignn.data.records import ScaffoldRecord


@dataclass(frozen=True)
class PhaseRegion:
    phase: int
    fraction: float
    density: float
    pore: bool


def allocate_regions(fractions: tuple[float, ...], minimum_nodes: int = 8, maximum_nodes: int = 25) -> list[PhaseRegion]:
    nodes = max(minimum_nodes, min(maximum_nodes, round(16 + 8 * len(fractions))))
    regions: list[PhaseRegion] = []
    for phase, fraction in enumerate(fractions):
        count = max(1, round(nodes * fraction))
        regions.extend(PhaseRegion(phase, fraction, 1.0, False) for _ in range(count))
    return regions[:maximum_nodes]


def connect_regions(regions: list[PhaseRegion], interpenetrating: bool) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    for source in range(len(regions)):
        for target in range(source + 1, len(regions)):
            same = regions[source].phase == regions[target].phase
            adjacent = abs(source - target) <= 2
            if adjacent and (same or interpenetrating):
                edges.extend(((source, target), (target, source)))
    if not edges:
        edges = [(0, 0)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_literature_graph(record: ScaffoldRecord) -> Data:
    fractions = record.composition + (record.porosity,)
    total = sum(fractions)
    normalized = tuple(value / total for value in fractions)
    regions = allocate_regions(normalized)
    interpenetrating = "interpenetrating" in record.architecture.lower()
    edge_index = connect_regions(regions, interpenetrating)
    features: list[list[float]] = []
    for region in regions:
        one_hot = [0.0] * 6
        one_hot[min(region.phase, 5)] = 1.0
        features.append(one_hot + [region.fraction, record.porosity, record.pore_size, record.density, record.solid_density, record.sintering_temperature, float(region.pore), 0.0, 0.0, 0.0, 0.0, 0.0])
    edge_features = torch.zeros((edge_index.shape[1], 8), dtype=torch.float32)
    edge_features[:, 0] = 1.0
    global_features = torch.tensor([[record.porosity, *record.composition[:4], record.pore_size, record.density, record.solid_density, record.sintering_temperature, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[record.compressive_strength, record.elastic_modulus, record.yield_strength]], dtype=torch.float32)
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_features, u=global_features, y=targets)
