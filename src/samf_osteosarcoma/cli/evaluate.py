import argparse
import json
from pathlib import Path

import numpy as np

from samf_osteosarcoma.metrics.survival import concordance_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="samf-evaluate")
    parser.add_argument("--predictions", type=Path, required=True)
    arguments = parser.parse_args()
    values = np.load(arguments.predictions)
    score = concordance_index(values["duration"], values["risk"], values["event"])
    result = {"c_index": score}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

