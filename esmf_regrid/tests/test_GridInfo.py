"""Unit tests for :class:`esmf_regrid.esmf_regridder.GridInfo`."""

import os

from esmf_regrid.esmf_regridder import GridInfo

import numpy as np


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
    """Basic test for :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
    lon, lat, lon_bounds, lat_bounds = _make_small_grid_args()
    grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    esmf_grid = grid.make_esmf_field()
    esmf_grid.data[:] = 0

    relative_path = os.path.join("results", "test_GridInfo", "small_grid.txt")
    abs_path = os.path.join(os.path.dirname(__file__), relative_path)
    with open(abs_path) as file:
        expected_repr = file.read()

    assert esmf_grid.__repr__() == expected_repr
