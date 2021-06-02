"""Unit tests for :func:`esmf_regrid.schemes.ESMFAreaWeighted`."""

import numpy as np
from numpy import ma
import pytest

from esmf_regrid.schemes import ESMFAreaWeighted
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube


def test_cube_regrid():
    """
    Test that ESMFAreaWeighted can be passed to a cubes regrid method.

    Checks that regridding occurs and that mdtol is used correctly.
    """
    scheme_default = ESMFAreaWeighted()
    scheme_full_mdtol = ESMFAreaWeighted(mdtol=1)

    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
    src_data = np.zeros([n_lats_src, n_lons_src])
    src_mask = np.zeros([n_lats_src, n_lons_src])
    src_mask[0, 0] = 1
    src_data = ma.array(src_data, mask=src_mask)
    src.data = src_data

    result_default = src.regrid(tgt, scheme_default)
    result_full = src.regrid(tgt, scheme_full_mdtol)

    expected_data_default = np.zeros([n_lats_tgt, n_lons_tgt])
    expected_mask = np.zeros([n_lats_tgt, n_lons_tgt])
    expected_mask[0, 0] = 1
    expected_data_full = ma.array(expected_data_default, mask=expected_mask)

    expected_cube_default = tgt.copy()
    expected_cube_default.data = expected_data_default

    expected_cube_full = tgt.copy()
    expected_cube_full.data = expected_data_full

    assert expected_cube_default == result_default
    assert expected_cube_full == result_full


def test_invalid_mdtol():
    """
    Test initialisation of :func:`esmf_regrid.schemes.ESMFAreaWeighted`.

    Checks that an error is raised when mdtol is out of range.
    """
    match = "Value for mdtol must be in range 0 - 1, got "
    with pytest.raises(ValueError, match=match):
        _ = ESMFAreaWeighted(mdtol=2)
    with pytest.raises(ValueError, match=match):
        _ = ESMFAreaWeighted(mdtol=-1)
