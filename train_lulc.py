"""Train the land-cover classifier from WorldCover labels. Run manually.

    .venv/bin/python train_lulc.py

Training sites and validation sites are disjoint. That matters more than the
usual random split: neighbouring pixels in one scene are strongly correlated, so
holding out random pixels from the same scenes would report an accuracy the model
cannot reproduce anywhere else. Spatially independent validation is the only kind
worth quoting to a customer.

The artifact records the feature order and the validation confusion matrix, so
inference can refuse a mismatched model and area estimates can be error-adjusted.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from lulc_sample import lulc, s2, worldcover

# Disjoint by island group as well as by box, so validation is not merely a
# different rectangle in the same landscape.
TRAIN_SITES = {
    "kalimantan_forest":  (114.50, -2.60, 114.75, -2.35),
    "kalimantan_coast":   (117.10, -0.70, 117.35, -0.45),
    "java_urban":         (107.55, -6.98, 107.80, -6.73),
    "java_crop":          (110.30, -7.20, 110.55, -6.95),
    "sumatra_plantation": (101.40,  0.30, 101.65,  0.55),
    "sulawesi_coast":     (119.35, -5.25, 119.60, -5.00),
}
VALID_SITES = {
    "papua_forest":       (138.50, -3.20, 138.75, -2.95),
    "sumatra_wetland":    (104.30, -2.55, 104.55, -2.30),
    "java_east_mixed":    (112.60, -7.40, 112.85, -7.15),
}

YEAR_RANGE = "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z"
PER_CLASS_PER_SITE = 4000
MODEL_PATH = Path("models/lulc_rf_v1.joblib")


async def sample_site(name: str, bbox: tuple[float, float, float, float]):
    """Feature rows and WorldCover labels for the clearest 2021 scene at a site."""
    items = await s2.stac_search(bbox, YEAR_RANGE, limit=40)
    if not items:
        print(f"  {name}: no scene"); return None
    items = [i for i in items if s2.scene_cloud(i) < 20]
    if not items:
        print(f"  {name}: nothing under 20% cloud"); return None
    item = min(items, key=s2.scene_cloud)

    scene = s2.read_scene_lulc(item, bbox)
    features, valid = lulc.stack_features(scene)
    if features.shape[0] < lulc.MIN_TRAIN_PIXELS:
        print(f"  {name}: too few valid pixels"); return None

    # Use the rounded Sentinel raster bounds, not the requested bbox, so the
    # reference labels and feature pixels describe the same grid footprint.
    codes = worldcover.read_labels(tuple(scene["bounds"]), valid.shape)
    labels = worldcover.to_index(codes)[valid]

    keep_x, keep_y = [], []
    for index in range(len(worldcover.CLASS_CODES)):
        rows = np.flatnonzero(labels == index)
        if rows.size == 0:
            continue
        if rows.size > PER_CLASS_PER_SITE:
            rows = np.random.default_rng(42).choice(rows, PER_CLASS_PER_SITE, replace=False)
        keep_x.append(features[rows]); keep_y.append(labels[rows])
    if not keep_x:
        print(f"  {name}: no labelled pixels"); return None

    x = np.vstack(keep_x); y = np.concatenate(keep_y)
    present = sorted({worldcover.CLASS_NAMES[worldcover.INDEX_TO_CODE[i]] for i in np.unique(y)})
    print(f"  {name}: {x.shape[0]:>6} px  cloud {s2.scene_cloud(item):.1f}%  {', '.join(present)}")
    return x, y


async def collect(sites: dict[str, tuple[float, float, float, float]]):
    xs, ys = [], []
    for name, bbox in sites.items():
        got = await sample_site(name, bbox)
        if got:
            xs.append(got[0]); ys.append(got[1])
    if not xs:
        raise SystemExit("no training data collected")
    return np.vstack(xs), np.concatenate(ys)


async def main() -> None:
    print("training sites:")
    x_train, y_train = await collect(TRAIN_SITES)
    print("validation sites (disjoint):")
    x_valid, y_valid = await collect(VALID_SITES)

    print(f"\ntrain {x_train.shape[0]} px / valid {x_valid.shape[0]} px "
          f"/ {x_train.shape[1]} features")

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    model.fit(x_train, y_train)

    predicted = model.predict(x_valid)
    n_classes = len(worldcover.CLASS_CODES)
    confusion = lulc.confusion_matrix(y_valid, predicted, n_classes)
    overall = float((predicted == y_valid).mean())

    print(f"\nspatially independent overall accuracy: {overall * 100:.1f}%")
    print(f"{'class':<20}{'n':>7}{'user acc':>10}")
    for index in range(n_classes):
        support = int(confusion[:, index].sum())
        if not support:
            continue
        mapped = int(confusion[index].sum())
        users = confusion[index][index] / mapped if mapped else float("nan")
        name = worldcover.CLASS_NAMES[worldcover.INDEX_TO_CODE[index]]
        print(f"{name:<20}{support:>7}{users * 100:>9.1f}%")

    importance = sorted(zip(lulc.FEATURE_ORDER, model.feature_importances_),
                        key=lambda kv: -kv[1])
    print("\ntop features: " + ", ".join(f"{k} {v:.3f}" for k, v in importance[:5]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({
        "model": model,
        "feature_order": lulc.FEATURE_ORDER,
        "class_codes": worldcover.CLASS_CODES,
        "confusion": confusion.tolist(),
        "overall_accuracy": overall,
        "label_source": "ESA WorldCover 2021 v200",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_sites": list(TRAIN_SITES), "valid_sites": list(VALID_SITES),
    }, MODEL_PATH)
    print(f"\nsaved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({"overall_accuracy": round(overall, 4)}))


if __name__ == "__main__":
    asyncio.run(main())
