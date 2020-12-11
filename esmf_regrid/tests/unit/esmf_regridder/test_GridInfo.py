# -*- coding: utf-8 -*-
"""Unit tests for :class:`esmf_regrid.esmf_regridder.GridInfo`."""

import ESMF
import iris
import numpy as np
import numpy.testing as nt

from esmf_regrid.esmf_regridder import GridInfo
import esmf_regrid.tests as tests


ESMF_LON, ESMF_LAT = 0, 1


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
    grid = GridInfo.from_1d_coords(lon, lat, lon_bounds, lat_bounds)
    esmf_grid = grid.make_esmf_field()
    esmf_grid.data[:] = 0

    relative_path = ("esmf_regridder", "test_GridInfo", "small_grid.txt")
    fname = tests.get_result_path(relative_path)
    with open(fname) as fi:
        expected_repr = fi.read()

    print(esmf_grid.__repr__())

    assert esmf_grid.__repr__() == expected_repr


def _simple_1d_bounds_to_coord(simple_bounds, *args, **kwargs):
    points = (simple_bounds[:-1] + simple_bounds[1:]) / 2.
    bounds = np.stack([simple_bounds[:-1], simple_bounds[1:]], axis=-1)
    return iris.coords.DimCoord(points, bounds=bounds, *args, **kwargs)


def test_from_cube_1d():
    lat_bounds_simple = np.linspace(-90., 90., 181)
    lat_points_simple = (lat_bounds_simple[:-1] + lat_bounds_simple[1:]) / 2.
    lat = _simple_1d_bounds_to_coord(
        lat_bounds_simple,
        standard_name='latitude',
        long_name='latitude',
        var_name='lat',
        units='degrees_north',
    )
    lon_bounds_simple = np.linspace(0., 360., 361)
    lon_points_simple = (lon_bounds_simple[:-1] + lon_bounds_simple[1:]) / 2.
    lon = _simple_1d_bounds_to_coord(
        lon_bounds_simple,
        standard_name='longitude',
        long_name='longitude',
        var_name='lon',
        units='degrees_north',
    )
    data = np.zeros(lat.shape + lon.shape)
    cube = iris.cube.Cube(
        data,
        dim_coords_and_dims=[(lat, 0), (lon, 1)]
    )
    grid_info = GridInfo.from_cube(cube)

    lon_points_expected, lat_points_expected = np.meshgrid(lon_points_simple,
                                                           lat_points_simple)
    lon_points = grid_info.grid.get_coords(ESMF_LON, ESMF.StaggerLoc.CENTER)
    lat_points = grid_info.grid.get_coords(ESMF_LAT, ESMF.StaggerLoc.CENTER)
    nt.assert_allclose(lon_points, lon_points_expected)
    nt.assert_allclose(lat_points, lat_points_expected)

    lon_bounds_expected, lat_bounds_expected = np.meshgrid(lon_bounds_simple,
                                                           lat_bounds_simple)
    lon_bounds = grid_info.grid.get_coords(ESMF_LON, ESMF.StaggerLoc.CORNER)
    lat_bounds = grid_info.grid.get_coords(ESMF_LAT, ESMF.StaggerLoc.CORNER)
    nt.assert_allclose(lon_bounds, lon_bounds_expected)
    nt.assert_allclose(lat_bounds, lat_bounds_expected)
