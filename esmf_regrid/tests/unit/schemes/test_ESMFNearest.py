"""Unit tests for :class:`esmf_regrid.schemes.ESMFNearest`."""

import numpy as np
from numpy import ma
import pytest

from esmf_regrid.schemes import ESMFNearest
from esmf_regrid.tests.unit.schemes.__init__ import (
    _test_mask_from_init,
    _test_mask_from_regridder,
    _test_non_degree_crs,
)
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
    _gridlike_mesh_cube,
)


@pytest.mark.parametrize(
    "src_type,tgt_type",
    [
        ("grid", "grid"),
        ("grid", "mesh"),
        ("grid", "just_mesh"),
        ("mesh", "grid"),
        ("mesh", "mesh"),
    ],
)
def test_cube_regrid(src_type, tgt_type):
    """
    Test that ESMFNearest can be passed to a cubes regrid method.

    Checks that regridding occurs.
    """
    if tgt_type == "just_mesh":
        scheme_default = ESMFNearest(tgt_location="face")
    else:
        scheme_default = ESMFNearest()

    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    if src_type == "grid":
        src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
        src_data = np.zeros([n_lats_src, n_lons_src])
        src_mask = np.zeros([n_lats_src, n_lons_src])
        src_mask[0, 0] = 1
    else:
        src = _gridlike_mesh_cube(n_lons_src, n_lats_src)
        src_data = np.zeros([n_lats_src * n_lons_src])
        src_mask = np.zeros([n_lats_src * n_lons_src])
        src_mask[0] = 1
    if tgt_type == "grid":
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
        expected_data_default = np.zeros([n_lats_tgt, n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt, n_lons_tgt])
        expected_mask[0, 0] = 1
    elif tgt_type == "mesh":
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
        expected_data_default = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask[0] = 1
    elif tgt_type == "just_mesh":
        tgt = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        expected_data_default = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask[0] = 1
    src_data = ma.array(src_data, mask=src_mask)
    src.data = src_data

    result_default = src.regrid(tgt, scheme_default)

    if tgt_type == "just_mesh":
        expected_cube_default = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
    else:
        expected_cube_default = tgt.copy()
    expected_cube_default.data = expected_data_default

    assert expected_cube_default == result_default


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_init(mask_keyword):
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFNearest`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_init(ESMFNearest, mask_keyword)


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_regridder(mask_keyword):
    """
    Test regridder method of :class:`esmf_regrid.schemes.ESMFNearest`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_regridder(ESMFNearest, mask_keyword)


def test_non_degree_crs():
    """Test for coordinates with non-degree units."""
    _test_non_degree_crs(ESMFNearest)
