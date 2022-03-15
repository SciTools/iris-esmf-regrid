"""Unit tests for :class:`esmf_regrid.esmf_regridder.Regridder`."""

import numpy as np
from numpy import ma
import pytest
import scipy.sparse

from esmf_regrid.esmf_regridder import GridInfo, Regridder
from esmf_regrid.tests import make_grid_args


def _expected_weights():
    weight_list = np.array(
        [
            0.6674194025656819,
            0.3325805974343169,
            0.3351257294386341,
            0.6648742705613656,
            0.33363933739884066,
            0.1663606626011589,
            0.333639337398841,
            0.1663606626011591,
            0.16742273275056854,
            0.33250863479149745,
            0.16742273275056876,
            0.33250863479149767,
            0.6674194025656823,
            0.3325805974343174,
            0.3351257294386344,
            0.6648742705613663,
        ]
    )
    rows = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5])
    columns = np.array([0, 1, 1, 2, 0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 4, 5])

    shape = (6, 6)

    weights = scipy.sparse.csr_matrix((weight_list, (rows, columns)), shape=shape)
    return weights


def test_Regridder_init():
    """Basic test for :meth:`~esmf_regrid.esmf_regridder.Regridder.__init__`."""
    lon, lat, lon_bounds, lat_bounds = make_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    lon, lat, lon_bounds, lat_bounds = make_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    rg = Regridder(src_grid, tgt_grid)

    result = rg.weight_matrix
    expected = _expected_weights()

    assert np.allclose(result.toarray(), expected.toarray())


def test_Regridder_init_fail():
    """Basic test for :meth:`~esmf_regrid.esmf_regridder.Regridder.__init__`."""
    lon, lat, lon_bounds, lat_bounds = make_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    lon, lat, lon_bounds, lat_bounds = make_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    with pytest.raises(ValueError):
        _ = Regridder(src_grid, tgt_grid, method="other")


def test_Regridder_regrid():
    """Basic test for :meth:`~esmf_regrid.esmf_regridder.Regridder.regrid`."""
    lon, lat, lon_bounds, lat_bounds = make_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    lon, lat, lon_bounds, lat_bounds = make_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    # Set up the regridder with precomputed weights.
    rg = Regridder(src_grid, tgt_grid, precomputed_weights=_expected_weights())

    src_array = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    src_masked = ma.array(src_array, mask=np.array([[1, 0], [0, 0], [0, 0]]))

    # Regrid with unmasked data.
    result_nomask = rg.regrid(src_array)
    expected_nomask = ma.array(
        [
            [1.0, 0.8336393373988409, 0.6674194025656824],
            [1.0, 0.4999999999999997, 0.0],
        ]
    )
    assert ma.allclose(result_nomask, expected_nomask)

    # Regrid with an masked array with no masked points.
    result_ma_nomask = rg.regrid(ma.array(src_array))
    assert ma.allclose(result_ma_nomask, expected_nomask)

    # Regrid with a fully masked array.
    result_fullmask = rg.regrid(ma.array(src_array, mask=True))
    expected_fulmask = ma.array(np.zeros([2, 3]), mask=True)
    assert ma.allclose(result_fullmask, expected_fulmask)

    # Regrid with a masked array containing a masked point.
    result_withmask = rg.regrid(src_masked)
    expected_withmask = ma.array(
        [
            [0.9999999999999999, 0.7503444126612077, 0.6674194025656824],
            [1.0, 0.4999999999999997, 0.0],
        ]
    )
    assert ma.allclose(result_withmask, expected_withmask)

    # Regrid while setting mdtol.
    result_half_mdtol = rg.regrid(src_masked, mdtol=0.5)
    expected_half_mdtol = ma.array(
        expected_withmask, mask=np.array([[1, 0, 1], [0, 0, 0]])
    )
    assert ma.allclose(result_half_mdtol, expected_half_mdtol)

    # Regrid with norm_type="dstarea".
    result_dstarea = rg.regrid(src_masked, norm_type="dstarea")
    expected_dstarea = ma.array(
        [
            [0.3325805974343169, 0.4999999999999999, 0.6674194025656823],
            [0.9999999999999998, 0.499931367542066, 0.0],
        ]
    )
    assert ma.allclose(result_dstarea, expected_dstarea)

    def _give_extra_dims(array):
        result = np.stack([array, array + 1])
        result = np.stack([result, result + 10, result + 100])
        return result

    # Regrid with multiple extra dimensions.
    extra_dim_src = _give_extra_dims(src_array)
    extra_dim_expected = _give_extra_dims(expected_nomask)

    extra_dim_result = rg.regrid(extra_dim_src)
    assert ma.allclose(extra_dim_result, extra_dim_expected)

    # Regrid extra dimensions with different masks.
    mixed_mask_src = ma.stack([src_array, src_masked])
    mixed_mask_expected = np.stack([expected_nomask, expected_withmask])

    mixed_mask_result = rg.regrid(mixed_mask_src)
    assert ma.allclose(mixed_mask_result, mixed_mask_expected)

    assert src_array.T.shape != src_array.shape
    with pytest.raises(ValueError):
        _ = rg.regrid(src_array.T)

    with pytest.raises(ValueError):
        _ = rg.regrid(src_masked, norm_type="INVALID")


def test_Regridder_init_small():
    """
    Simplified test for :meth:`~esmf_regrid.esmf_regridder.Regridder.regrid`.

    This test is designed to be simple enough to generate predictable weights.
    With predictable weights it is easier to check that the weights are
    associated with the correct indices.
    """
    # The following ASCII visualisation describes the source and target grids
    # and the indices which ESMF assigns to their cells.
    # Analysis of the weights dict returned by ESMF confirms that this is
    # indeed the indexing which ESMF assigns these grids.
    #
    # 30  +---+---+       +-------+
    #     | 3 | 6 |       |       |
    # 20  +---+---+       |   2   |
    #     | 2 | 5 |       |       |
    # 10  +---+---+       +-------+
    #     | 1 | 4 |       |   1   |
    #  0  +---+---+       +-------+
    #     0  10  20       0       20
    def _get_points(bounds):
        points = (bounds[:-1] + bounds[1:]) / 2
        return points

    lon_bounds = np.array([0, 10, 20])
    lat_bounds = np.array([0, 10, 20, 30])
    lon, lat = _get_points(lon_bounds), _get_points(lat_bounds)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    assert src_grid.shape == (3, 2)
    assert src_grid.index_offset == 1

    lon_bounds = np.array([0, 20])
    lat_bounds = np.array([0, 10, 30])
    lon, lat = _get_points(lon_bounds), _get_points(lat_bounds)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    assert tgt_grid.shape == (2, 1)
    assert tgt_grid.index_offset == 1

    rg = Regridder(src_grid, tgt_grid)

    result = rg.weight_matrix

    weights_dict = {}
    weights_dict["row_dst"] = (
        np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32) - src_grid.index_offset
    )
    weights_dict["col_src"] = (
        np.array([1, 2, 4, 5, 2, 3, 5, 6], dtype=np.int32) - tgt_grid.index_offset
    )
    # The following weights are calculated from grids on a sphere with great circles
    # defining the lines. Because of this, weights are not exactly the same as they
    # would be in euclidean geometry and there are some small additional weights
    # where the cells would not ordinarily overlap in a euclidean grid.
    weights_dict["weights"] = np.array(
        [
            0.4962897,
            0.0037103,  # Small weight due to using great circle lines
            0.4962897,
            0.0037103,  # Small weight due to using great circle lines
            0.25484249,
            0.24076863,
            0.25484249,
            0.24076863,
        ]
    )

    expected_weights = scipy.sparse.csr_matrix(
        (weights_dict["weights"], (weights_dict["row_dst"], weights_dict["col_src"]))
    )
    assert np.allclose(result.toarray(), expected_weights.toarray())
