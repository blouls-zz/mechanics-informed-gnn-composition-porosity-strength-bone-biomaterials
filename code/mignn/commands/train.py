import argparse
import logging
from pathlib import Path

from mignn.config import load_configuration


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="mignn-train")
    value.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    configuration = load_configuration(arguments.config)
    logging.getLogger(__name__).info("loaded experiment %s", configuration["experiment"])


if __name__ == "__main__":
    main()
