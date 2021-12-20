"""Unit tests for :class:`esmf_regrid._esmf_sdo.GridInfo`."""

import numpy as np
import pytest

from esmf_regrid._esmf_sdo import GridInfo
import esmf_regrid.tests as tests


def _make_small_grid_args():
    small_x = 2
    small_y = 3
    small_grid_lon = np.array(range(small_x)) / (small_x + 1)
    small_grid_lat = np.array(range(small_y)) * 2 / (small_y + 1)

    small_grid_lon_bounds = np.array(range(small_x + 1)) / (small_x + 1)
    small_grid_lat_bounds = np.array(range(small_y + 1)) * 2 / (small_y + 1)
    return (
        small_grid_lon,
        small_grid_lat,
        small_grid_lon_bounds,
        small_grid_lat_bounds,
    )


def test_make_grid():
    """Basic test for :meth:`~esmf_regrid._esmf_sdo.GridInfo.make_esmf_field`."""
    lon, lat, lon_bounds, lat_bounds = _make_small_grid_args()
    grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    esmf_grid = grid.make_esmf_field()
    esmf_grid.data[:] = 0

    relative_path = ("_esmf_sdo", "test_GridInfo", "small_grid.txt")
    fname = tests.get_result_path(relative_path)
    with open(fname) as fi:
        expected_repr = fi.read()

    assert esmf_grid.__repr__() == expected_repr


def test_GridInfo_init_fail():
    """
    Basic test for :meth:`~esmf_regrid.esmf_regridder.Regridder.__init__`.

    Tests that appropriate errors are raised for invalid data.
    """
    latlon_1D = np.ones(3)
    latlon_2D = np.ones([3, 3])
    latlon_3D = np.ones([3, 3, 3])

    with pytest.raises(ValueError) as excinfo:
        _ = GridInfo(latlon_1D, latlon_1D, latlon_2D, latlon_1D)
    expected_message = (
        "The dimensionality of longitude bounds "
        "(2) is incompatible with the "
        "dimensionality of the longitude (1)."
    )
    assert expected_message in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = GridInfo(latlon_1D, latlon_1D, latlon_1D, latlon_2D)
    expected_message = (
        "The dimensionality of latitude bounds "
        "(2) is incompatible with the "
        "dimensionality of the latitude (1)."
    )
    assert expected_message in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = GridInfo(latlon_1D, latlon_2D, latlon_1D, latlon_2D)
    expected_message = (
        "The dimensionality of the longitude "
        "(1) is incompatible with the "
        "dimensionality of the latitude (2)."
    )
    assert expected_message in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = GridInfo(latlon_3D, latlon_3D, latlon_3D, latlon_3D)
    expected_message = (
        "Expected a latitude/longitude with a dimensionality of 1 or 2, got 3."
    )
    assert expected_message in str(excinfo.value)
