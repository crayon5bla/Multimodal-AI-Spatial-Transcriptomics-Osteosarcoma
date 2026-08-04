import argparse
from pathlib import Path

from samf_osteosarcoma.data.manifests import validate_spatial_manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="samf-prepare")
    parser.add_argument("--spatial-manifest", type=Path, required=True)
    arguments = parser.parse_args()
    validate_spatial_manifest(arguments.spatial_manifest)


if __name__ == "__main__":
    main()

