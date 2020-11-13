"""Unit tests for :class:`esmf_regrid.esmf_regridder.Regridder`."""

import numpy as np

from esmf_regrid.esmf_regridder import GridInfo, Regridder
import esmf_regrid.tests as tests
from numpy import ma
import scipy.sparse


def make_small_grid_args(x, y):
    small_grid_lon = np.array(range(x)) * 10 / x
    small_grid_lat = np.array(range(y)) * 10 / y

    small_grid_lon_bounds = np.array(range(x + 1)) * 10 / x
    small_grid_lat_bounds = np.array(range(y + 1)) * 10 / y
    return (
        small_grid_lon,
        small_grid_lat,
        small_grid_lon_bounds,
        small_grid_lat_bounds,
    )


def expected_weights():
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
    lon, lat, lon_bounds, lat_bounds = make_small_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    lon, lat, lon_bounds, lat_bounds = make_small_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    rg = Regridder(src_grid, tgt_grid)

    result = rg.weight_matrix
    expected = expected_weights()

    assert np.allclose(result.toarray(), expected.toarray())
