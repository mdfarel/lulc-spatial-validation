"""Shared feature and evaluation contracts for the optional LULC trainer."""

from __future__ import annotations

import numpy as np


FEATURE_ORDER = (
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndbi", "bsi", "ndmi",
)
MIN_TRAIN_PIXELS = 1_000


def stack_features(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return finite valid pixels as rows in the locked feature order."""
    missing = [name for name in FEATURE_ORDER if name not in scene]
    if missing:
        raise ValueError(f"land-cover scene is missing features: {', '.join(missing)}")

    shapes = [np.asarray(scene[name]).shape for name in FEATURE_ORDER]
    shapes.append(np.asarray(scene["valid"]).shape)
    if len(set(shapes)) != 1:
        raise ValueError("land-cover feature rasters are not aligned")

    cube = np.stack([np.asarray(scene[name], dtype=np.float32) for name in FEATURE_ORDER], axis=-1)
    valid = np.asarray(scene["valid"], dtype=bool) & np.all(np.isfinite(cube), axis=-1)
    return cube[valid], valid


def confusion_matrix(reference: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are mapped/predicted classes; columns are reference classes.

    That orientation makes row-normalised diagonal values user's accuracy and
    column-normalised diagonal values producer's accuracy.
    """
    reference = np.asarray(reference, dtype=np.int64).ravel()
    predicted = np.asarray(predicted, dtype=np.int64).ravel()
    if reference.shape != predicted.shape:
        raise ValueError("reference and predicted labels must have the same shape")
    valid = (
        (reference >= 0) & (reference < n_classes)
        & (predicted >= 0) & (predicted < n_classes)
    )
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (predicted[valid], reference[valid]), 1)
    return matrix
