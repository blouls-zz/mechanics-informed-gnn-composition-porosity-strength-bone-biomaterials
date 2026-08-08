from collections.abc import Callable
from math import cos, pi, sin

import torch
from torch_geometric.data import Data


def gyroid(x: float, y: float, z: float) -> float:
    return sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


def schwarz_p(x: float, y: float, z: float) -> float:
    return cos(x) + cos(y) + cos(z)


def diamond(x: float, y: float, z: float) -> float:
    return sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) + cos(x) * cos(y) * sin(z)


def iw_p(x: float, y: float, z: float) -> float:
    return 2.0 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x)) - (cos(2.0 * x) + cos(2.0 * y) + cos(2.0 * z))


SURFACES: dict[str, Callable[[float, float, float], float]] = {"gyroid": gyroid, "schwarz_p": schwarz_p, "diamond": diamond, "i_wp": iw_p}


def voxel_graph(architecture: str, level: float, material: tuple[float, ...], grid: int = 5) -> Data:
    surface = SURFACES[architecture]
    occupied: list[tuple[int, int, int]] = []
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                coordinates = tuple(2.0 * pi * value / grid for value in (i, j, k))
                if surface(*coordinates) >= level:
                    occupied.append((i, j, k))
    lookup = {position: index for index, position in enumerate(occupied)}
    edges: list[tuple[int, int]] = []
    for index, (i, j, k) in enumerate(occupied):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    neighbor = ((i + di) % grid, (j + dj) % grid, (k + dk) % grid)
                    if neighbor in lookup and neighbor != (i, j, k):
                        edges.append((index, lookup[neighbor]))
    node_features = torch.zeros((len(occupied), 18), dtype=torch.float32)
    node_features[:, : min(len(material), 6)] = torch.tensor(material[:6])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_features = torch.zeros((len(edges), 8), dtype=torch.float32)
    edge_features[:, 3] = 1.0
    state = torch.zeros((1, 10), dtype=torch.float32)
    state[0, 0] = 1.0 - len(occupied) / grid**3
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, u=state)
