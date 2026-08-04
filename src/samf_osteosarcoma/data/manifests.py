import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


REQUIRED_SPATIAL_FIELDS = {
    "patient_id",
    "slide_id",
    "spot_id",
    "image_path",
    "matrix_path",
    "x",
    "y",
    "scalefactors_path",
}


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def validate_spatial_manifest(path: Path) -> int:
    rows = list(read_rows(path))
    if not rows:
        raise ValueError("spatial manifest is empty")
    missing = REQUIRED_SPATIAL_FIELDS.difference(rows[0])
    if missing:
        raise ValueError(f"spatial manifest missing fields: {sorted(missing)}")
    patients = {row["patient_id"] for row in rows}
    if any(not row["scalefactors_path"] for row in rows):
        raise ValueError("Space Ranger scale factors are required")
    return len(patients)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest_metadata(source: Path, destination: Path) -> None:
    payload: dict[str, Any] = {
        "file": source.name,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)

