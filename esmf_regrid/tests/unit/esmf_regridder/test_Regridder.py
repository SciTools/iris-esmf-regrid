"""Unit tests for :class:`esmf_regrid.esmf_regridder.Regridder`."""

import numpy as np
from numpy import ma
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


def test_Regridder_regrid():
    """Basic test for :meth:`~esmf_regrid.esmf_regridder.Regridder.regrid`."""
    lon, lat, lon_bounds, lat_bounds = make_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    lon, lat, lon_bounds, lat_bounds = make_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    # Set up the regridder with precomputed weights.
    rg = Regridder(src_grid, tgt_grid, precomputed_weights=_expected_weights())

    src_array = np.array([[1, 1, 1], [1, 0, 0]])
    src_masked = ma.array(src_array, mask=[[1, 0, 0], [0, 0, 0]])

    # Regrid with unmasked data.
    result_nomask = rg.regrid(src_array)
    expected_nomask = ma.array(
        [
            [1.0, 1.0],
            [0.8336393373988409, 0.4999999999999997],
            [0.6674194025656824, 0.0],
        ]
    )
    assert ma.allclose(result_nomask, expected_nomask)

    # Regrid with an masked array with no masked points.
    result_ma_nomask = rg.regrid(ma.array(src_array))
    assert ma.allclose(result_ma_nomask, expected_nomask)

    # Regrid with a fully masked array.
    result_fullmask = rg.regrid(ma.array(src_array, mask=True))
    expected_fulmask = ma.array(np.zeros([3, 2]), mask=True)
    assert ma.allclose(result_fullmask, expected_fulmask)

    # Regrid with a masked array containing a masked point.
    result_withmask = rg.regrid(src_masked)
    expected_withmask = ma.array(
        [
            [0.9999999999999999, 1.0],
            [0.7503444126612077, 0.4999999999999997],
            [0.6674194025656824, 0.0],
        ]
    )
    assert ma.allclose(result_withmask, expected_withmask)

    # Regrid while setting mdtol.
    result_half_mdtol = rg.regrid(src_masked, mdtol=0.5)
    expected_half_mdtol = ma.array(expected_withmask, mask=[[1, 0], [0, 0], [1, 0]])
    assert ma.allclose(result_half_mdtol, expected_half_mdtol)

    # Regrid with norm_type="DSTAREA".
    result_dstarea = rg.regrid(src_masked, norm_type="DSTAREA")
    expected_dstarea = ma.array(
        [
            [0.3325805974343169, 0.9999999999999998],
            [0.4999999999999999, 0.499931367542066],
            [0.6674194025656823, 0.0],
        ]
    )
    assert ma.allclose(result_dstarea, expected_dstarea)
