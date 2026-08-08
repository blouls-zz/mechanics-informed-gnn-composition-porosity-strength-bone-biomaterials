from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from mignn.data.records import ScaffoldRecord
from mignn.types import SplitFractions


def stratified_split(records: Sequence[ScaffoldRecord], fractions: SplitFractions, seed: int) -> tuple[list[int], list[int], list[int]]:
    if not np.isclose(fractions.train + fractions.validation + fractions.test, 1.0):
        raise ValueError("split fractions must sum to one")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        porosity_bin = min(int(record.porosity * 10.0), 9)
        groups[(record.family, porosity_bin)].append(index)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        train_end = round(len(shuffled) * fractions.train)
        validation_end = train_end + round(len(shuffled) * fractions.validation)
        train.extend(shuffled[:train_end].tolist())
        validation.extend(shuffled[train_end:validation_end].tolist())
        test.extend(shuffled[validation_end:].tolist())
    return sorted(train), sorted(validation), sorted(test)


def composition_ood_split(records: Sequence[ScaffoldRecord], held_out: set[str]) -> tuple[list[int], list[int]]:
    training: list[int] = []
    testing: list[int] = []
    for index, record in enumerate(records):
        target = testing if record.architecture in held_out or record.family in held_out else training
        target.append(index)
    return training, testing
