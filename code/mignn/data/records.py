from dataclasses import dataclass
from enum import IntEnum


class InterfaceType(IntEnum):
    CERAMIC_CERAMIC = 0
    CERAMIC_POLYMER = 1
    PHASE_PORE = 2
    METAL_PORE = 3
    POLYMER_PORE = 4


@dataclass(frozen=True)
class ScaffoldRecord:
    identifier: str
    family: str
    composition: tuple[float, ...]
    porosity: float
    pore_size: float
    density: float
    solid_density: float
    solid_strength: float
    sintering_temperature: float
    manufacturing_method: str
    architecture: str
    compressive_strength: float
    elastic_modulus: float
    yield_strength: float
    elastic_limit_strain: float
