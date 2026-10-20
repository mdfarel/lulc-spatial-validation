import numpy as np
import pytest

from lulc_sample import lulc, worldcover


def test_feature_stack_uses_locked_order_and_joint_validity():
    shape = (2, 3)
    scene = {
        name: np.full(shape, index, dtype=np.float32)
        for index, name in enumerate(lulc.FEATURE_ORDER)
    }
    scene["valid"] = np.ones(shape, dtype=bool)
    scene["ndvi"][0, 1] = np.nan

    features, valid = lulc.stack_features(scene)

    assert features.shape == (5, len(lulc.FEATURE_ORDER))
    assert not valid[0, 1]
    np.testing.assert_array_equal(features[0], np.arange(len(lulc.FEATURE_ORDER)))


def test_feature_stack_rejects_misaligned_rasters():
    scene = {name: np.ones((2, 2), dtype=np.float32) for name in lulc.FEATURE_ORDER}
    scene["valid"] = np.ones((3, 2), dtype=bool)
    with pytest.raises(ValueError, match="not aligned"):
        lulc.stack_features(scene)


def test_confusion_matrix_orientation_supports_user_and_producer_accuracy():
    reference = np.array([0, 0, 1, 1, 1])
    predicted = np.array([0, 1, 1, 1, 0])

    matrix = lulc.confusion_matrix(reference, predicted, 2)

    np.testing.assert_array_equal(matrix, [[1, 1], [1, 2]])
    assert matrix[1, 1] / matrix[1].sum() == pytest.approx(2 / 3)  # user's accuracy
    assert matrix[1, 1] / matrix[:, 1].sum() == pytest.approx(2 / 3)  # producer's accuracy


def test_worldcover_tile_naming_across_equator_and_prime_meridian():
    assert worldcover.tile_names((100.0, -1.0, 101.0, 1.0)) == ["S03E099", "N00E099"]
    assert worldcover.tile_names((-1.0, -1.0, 1.0, 1.0)) == [
        "S03W003", "S03E000", "N00W003", "N00E000",
    ]


def test_worldcover_codes_convert_to_dense_indices_and_preserve_unknown():
    codes = np.array([[10, 50, 95, 0, 255]], dtype=np.uint8)
    indices = worldcover.to_index(codes)
    assert indices.tolist() == [[0, 4, 9, -1, -1]]


def test_worldcover_href_matches_flat_bucket_layout():
    # The v200 bucket stores every tile directly under map/, with no per-tile
    # folder. A per-tile folder returns HTTP 404 for every tile.
    assert worldcover._href("S03E114") == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_S03E114_Map.tif"
    )
