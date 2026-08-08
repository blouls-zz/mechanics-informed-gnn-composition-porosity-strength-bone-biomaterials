import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="mignn-evaluate")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--targets", type=Path, required=True)
    return value


def main() -> None:
    parser().parse_args()


if __name__ == "__main__":
    main()
