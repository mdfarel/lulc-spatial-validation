# Land-cover classification with spatially independent validation

A small, self-contained sample from GEOAI Indonesia's Earth-observation engine
(Enviro). It trains a land-cover classifier for Indonesia from open Sentinel-2
imagery and ESA WorldCover labels, and reports its accuracy only on sites that
are geographically disjoint from the training sites.

Shared as a code sample for academic review.

## What it does

1. For each site, searches the Earth Search STAC API for the clearest 2021
   Sentinel-2 L2A scene (under 20% cloud) and reads a window on the 20 m grid.
2. Builds a 12-feature stack per pixel: six reflectance bands (blue, green,
   red, NIR, SWIR1, SWIR2) and six indices (NDVI, NDWI, MNDWI, NDBI, BSI,
   NDMI). Training and inference share one function, so the features cannot
   drift apart.
3. Reads ESA WorldCover 2021 labels on exactly the same grid footprint.
4. Samples up to 4,000 pixels per class per site and trains a random forest.
5. Validates on three sites in other islands and landscapes, and reports a
   confusion matrix and user's accuracy per class.

| Training sites | Validation sites (disjoint) |
|---|---|
| Kalimantan forest, Kalimantan coast, Java urban (Bandung), Java cropland, Sumatra plantation, Sulawesi coast | Papua forest, Sumatra wetland, East Java mixed |

## Why the split matters

Neighbouring pixels in one scene are strongly correlated. Holding out random
pixels from the training scenes reports an accuracy the model cannot
reproduce anywhere else. Validating on different islands is the honest test of
whether a model transfers, and it is the same question any model trained in
one landscape faces when it is applied to another.

## Results

Latest run, 20 October 2026 ([`results/training_run_2026-10-20.log`](results/training_run_2026-10-20.log)):
143,498 training pixels from six sites, 38,939 validation pixels from two
disjoint sites. The East Java validation site was skipped because its clearest
2021 scene had too few valid pixels.

| Class | Validation pixels | User's accuracy |
|---|---|---|
| Tree cover | 8,000 | 50.2% |
| Shrubland | 4,055 | 29.8% |
| Grassland | 8,000 | 35.0% |
| Cropland | 4,014 | 19.1% |
| Built-up | 4,000 | 63.6% |
| Bare/sparse vegetation | 399 | 6.8% |
| Permanent water | 5,689 | 60.0% |
| Herbaceous wetland | 4,627 | 38.6% |
| Mangroves | 155 | 2.1% |

**Overall accuracy on the disjoint sites: 35.6%.** The validation pixels are
sampled per class (up to 4,000 per class per site), so this is accuracy on a
class-stratified sample, not an area-weighted estimate. The most important
features were SWIR2, NDVI, NDWI, red and blue.

Earlier experiments with the same design (August 2026, before two pipeline
corrections described below) gave, on the disjoint validation sites:

| Experiment | Overall accuracy |
|---|---|
| 9 classes, single date | 28.1% |
| 9 classes, 4-season temporal stack | 39.5% |
| 4 coarse classes | 72.9% |
| Mangrove + oil palm | 44.3% (oil palm user's accuracy 37.8%) |

The conclusion was not to ship it. The 4-class model only matches what
WorldCover (76.7%) and Dynamic World (about 72%) already provide for free. The
mangrove and oil-palm run failed, and the products that succeed at those two
classes (Global Mangrove Watch, the Descals oil-palm map) are built on radar,
which this optical-only model does not use. A model trained on WorldCover
labels can also never be more accurate than WorldCover itself.

## Corrections made along the way

- **Sentinel-2 reflectance offset.** Since processing baseline 04.00 (January
  2022), L2A reflectance carries a -1000 DN offset. Handling it wrongly
  inflates every normalised-difference index, most for dark pixels such as
  water, shadow and dense canopy. See `_boa_offset` in `lulc_sample/s2.py`:
  the Earth Search COGs used here already have the offset removed even though
  their metadata still advertises it.
- **WorldCover tile URL.** A refactored helper looked for tiles in per-tile
  folders, which do not exist in the bucket, so every read returned HTTP 404.
  The unit tests passed because they never touched the network. Fixed on
  2026-10-20, with a regression test.

## Known limitations

- Single-date, optical-only features. Cloud limits which scenes are usable,
  and radar would be needed for mangrove and oil palm.
- The scene-classification mask removes pixels Sentinel-2 itself flags as
  water (SCL class 6), so the water class is learned only from water pixels
  that Sen2Cor did not flag, a biased sample.
- One validation site dropped out for cloud, so the disjoint test rests on two
  sites (Papua, Sumatra).
- Labels come from a global product, not field data, so the accuracy reported
  is agreement with WorldCover, not with the ground.

## How it was built

I designed the method, the site split and the validation, and judged the
results. The code was written with AI coding agents (Claude Code) working
under my direction, then run, tested and checked against the literature and
live data.

## Run it

Python 3.11 or later. All data are public and need no credentials.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python train_lulc.py
```

The trained model is written to `models/lulc_rf_v1.joblib` together with the
feature order, class codes, confusion matrix and site lists, so inference can
refuse a mismatched model.

## Layout

```
train_lulc.py            training and spatially independent validation
lulc_sample/s2.py        Sentinel-2 STAC search, windowed COG reads, feature stack
lulc_sample/lulc.py      feature-order contract and confusion matrix
lulc_sample/worldcover.py  ESA WorldCover 2021 labels on the scene grid
tests/                   unit tests (no network)
results/                 training run logs
```

Muhammad Dita Farel, GEOAI Indonesia
