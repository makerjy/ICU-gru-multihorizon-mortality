from __future__ import annotations

from .config import RunConfig
from .train import train_and_evaluate


def main() -> None:
    cfg = RunConfig()
    train_and_evaluate(cfg)


if __name__ == "__main__":
    main()
