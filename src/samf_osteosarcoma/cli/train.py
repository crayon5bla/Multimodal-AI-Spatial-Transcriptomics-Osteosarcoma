import argparse
import logging
from pathlib import Path

import torch

from samf_osteosarcoma.config import load_mapping
from samf_osteosarcoma.models.samf import SAMF
from samf_osteosarcoma.training.seed import set_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="samf-train")
    value.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    value.add_argument("--membership", type=Path, required=True)
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_mapping(arguments.config)
    set_seed(int(config["seed"]))
    membership = torch.load(arguments.membership, map_location="cpu", weights_only=True)
    model = SAMF(
        membership=membership,
        feature_dim=int(config["feature_dim"]),
        genes=int(config["genes"]),
    )
    parameters = sum(value.numel() for value in model.parameters())
    logging.info("initialized SAMF with %d trainable parameters", parameters)
    if not arguments.dry_run:
        raise RuntimeError("training requires prepared fold manifests")


if __name__ == "__main__":
    main()

