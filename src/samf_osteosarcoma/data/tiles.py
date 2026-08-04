from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileCoordinate:
    x: int
    y: int
    level: int
    size: int


def rgb_to_hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    array = rgb.astype(np.float32) / 255.0
    maximum = array.max(axis=-1)
    minimum = array.min(axis=-1)
    return np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    histogram, edges = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0))
    probability = histogram.astype(np.float64) / max(histogram.sum(), 1)
    omega = np.cumsum(probability)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(probability * centers)
    total = mu[-1]
    between = (total * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return float(centers[int(np.argmax(between))])


def tissue_fraction(tile: np.ndarray) -> float:
    saturation = rgb_to_hsv_saturation(tile)
    threshold = otsu_threshold(saturation)
    return float(np.mean(saturation >= threshold))


def tile_is_tissue(tile: np.ndarray) -> bool:
    saturation = rgb_to_hsv_saturation(tile)
    return float(saturation.mean()) >= 0.05 and tissue_fraction(tile) >= 0.5


def iter_coordinates(width: int, height: int, size: int = 256) -> Iterator[TileCoordinate]:
    for y in range(0, height - size + 1, size):
        for x in range(0, width - size + 1, size):
            yield TileCoordinate(x=x, y=y, level=0, size=size)


def extract_image_tiles(path: Path, output: Path, size: int = 256) -> list[Path]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for coordinate in iter_coordinates(image.width, image.height, size):
        tile = array[
            coordinate.y : coordinate.y + size,
            coordinate.x : coordinate.x + size,
        ]
        if tile_is_tissue(tile):
            destination = output / f"{coordinate.x}_{coordinate.y}.png"
            Image.fromarray(tile).save(destination)
            written.append(destination)
    return written

