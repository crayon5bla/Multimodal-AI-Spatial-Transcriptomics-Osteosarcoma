#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc_per_node=1 -m samf_osteosarcoma.cli.train --config configs/main.yaml --membership data/pathway_membership.pt

