#!/usr/bin/env bash
set -euo pipefail
python -m samf_osteosarcoma.cli.evaluate --predictions artifacts/target_os_predictions.npz

