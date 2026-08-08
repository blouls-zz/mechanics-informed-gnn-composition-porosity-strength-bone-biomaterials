from pathlib import Path
from typing import Any

import yaml


def load_configuration(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a mapping")
    return document
