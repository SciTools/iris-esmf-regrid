"""Unit tests for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`."""

import dask.array as da
from iris.coord_systems import RotatedGeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np
from numpy import ma

from esmf_regrid.schemes import regrid_rectilinear_to_rectilinear
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)


def test_rotated_regridding():
    """
    Test for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`.

    Test the regriding of a rotated pole coordinate system. The test is
    designed to that it should be possible to verify the result by
    inspection.
    """
    src_coord_system = RotatedGeogCS(0, 90, 90)
    tgt_coord_system = None

    n_lons = 4
    n_lats = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    src = _grid_cube(
        n_lons,
        n_lats,
        lon_bounds,
        lat_bounds,
        circular=True,
        coord_system=src_coord_system,
    )
    tgt = _grid_cube(
        n_lons,
        n_lats,
        lon_bounds,
        lat_bounds,
        circular=True,
        coord_system=tgt_coord_system,
    )
    src_data = np.arange(n_lons * n_lats).reshape([n_lats, n_lons])
    # src_mask = np.empty([n_lats, n_lons])
    # src_mask[:] = np.array([1, 0, 0, 1])[:, np.newaxis]
    src_mask = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    src_data = ma.array(src_data, mask=src_mask)
    src.data = src_data

    no_mdtol_result = regrid_rectilinear_to_rectilinear(src, tgt)

    full_mdtol_result = regrid_rectilinear_to_rectilinear(src, tgt, mdtol=1)

    expected_data = np.array(
        [[5, 4, 8, 9], [5, 4, 8, 9], [6, 7, 11, 10], [6, 7, 11, 10]]
    )
    expected_mask = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    no_mdtol_expected_data = ma.array(expected_data, mask=expected_mask)

    # Lenient check for data.
    assert np.allclose(no_mdtol_expected_data, no_mdtol_result.data)
    assert np.allclose(expected_data, full_mdtol_result.data)


def test_extra_dims():
    """
    Test for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`.

    Tests the handling of extra dimensions and metadata. Ensures that proper
    coordinates, attributes, names and units are copied over.
    """
    h = 2
    t = 4
    e = 6
    src_lats = 3
    src_lons = 5

    tgt_lats = 5
    tgt_lons = 3

    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    src_grid = _grid_cube(
        src_lons,
        src_lats,
        lon_bounds,
        lat_bounds,
    )
    tgt_grid = _grid_cube(
        tgt_lons,
        tgt_lats,
        lon_bounds,
        lat_bounds,
    )

    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")
    extra = AuxCoord(np.arange(e), long_name="extra dim")
    spanning = AuxCoord(np.ones([h, t, e]), long_name="spanning dim")

    src_data = np.empty([h, src_lats, t, src_lons, e])
    src_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    src_cube = Cube(src_data)
    src_cube.add_dim_coord(height, 0)
    src_cube.add_dim_coord(src_grid.coord("latitude"), 1)
    src_cube.add_dim_coord(time, 2)
    src_cube.add_dim_coord(src_grid.coord("longitude"), 3)
    src_cube.add_aux_coord(extra, 4)
    src_cube.add_aux_coord(spanning, [0, 2, 4])

    def _add_metadata(cube):
        result = cube.copy()
        result.units = "K"
        result.attributes = {"a": 1}
        result.standard_name = "air_temperature"
        scalar_height = AuxCoord([5], units="m", standard_name="height")
        scalar_time = DimCoord([10], units="s", standard_name="time")
        result.add_aux_coord(scalar_height)
        result.add_aux_coord(scalar_time)
        return result

    src_cube = _add_metadata(src_cube)

    result = regrid_rectilinear_to_rectilinear(src_cube, tgt_grid)

    expected_data = np.empty([h, tgt_lats, t, tgt_lons, e])
    expected_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(height, 0)
    expected_cube.add_dim_coord(tgt_grid.coord("latitude"), 1)
    expected_cube.add_dim_coord(time, 2)
    expected_cube.add_dim_coord(tgt_grid.coord("longitude"), 3)
    expected_cube.add_aux_coord(extra, 4)
    expected_cube.add_aux_coord(spanning, [0, 2, 4])
    expected_cube = _add_metadata(expected_cube)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and coords.
    result.data = expected_data
    assert expected_cube == result


def test_laziness():
    """Test that regridding is lazy when source data is lazy."""
    n_lons = 12
    n_lats = 10
    h = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    grid = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    src_data = np.arange(n_lats * n_lons * h).reshape([n_lats, n_lons, h])
    src_data = da.from_array(src_data, chunks=[3, 5, 1])
    src = Cube(src_data)
    src.add_dim_coord(grid.coord("latitude"), 0)
    src.add_dim_coord(grid.coord("longitude"), 1)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    assert src.has_lazy_data()
    result = regrid_rectilinear_to_rectilinear(src, tgt)
    assert result.has_lazy_data()
    assert np.allclose(result.data, src_data)


def test_extra_dims_curvilinear():
    """
    Test for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`.

    Tests the handling of extra dimensions and metadata. Ensures that proper
    coordinates, attributes, names and units are copied over.
    """
    h = 2
    t = 4
    e = 6
    src_lats = 3
    src_lons = 5

    tgt_lats = 5
    tgt_lons = 3

    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    src_grid = _curvilinear_cube(
        src_lons,
        src_lats,
        lon_bounds,
        lat_bounds,
    )
    tgt_grid = _curvilinear_cube(
        tgt_lons,
        tgt_lats,
        lon_bounds,
        lat_bounds,
    )

    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")
    extra = AuxCoord(np.arange(e), long_name="extra dim")
    spanning = AuxCoord(np.ones([h, t, e]), long_name="spanning dim")

    src_data = np.empty([h, src_lats, t, src_lons, e])
    src_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    src_cube = Cube(src_data)
    src_cube.add_dim_coord(height, 0)
    src_cube.add_aux_coord(src_grid.coord("latitude"), (1, 3))
    src_cube.add_dim_coord(time, 2)
    src_cube.add_aux_coord(src_grid.coord("longitude"), (1, 3))
    src_cube.add_aux_coord(extra, 4)
    src_cube.add_aux_coord(spanning, [0, 2, 4])

    def _add_metadata(cube):
        result = cube.copy()
        result.units = "K"
        result.attributes = {"a": 1}
        result.standard_name = "air_temperature"
        scalar_height = AuxCoord([5], units="m", standard_name="height")
        scalar_time = DimCoord([10], units="s", standard_name="time")
        result.add_aux_coord(scalar_height)
        result.add_aux_coord(scalar_time)
        return result

    src_cube = _add_metadata(src_cube)

    result = regrid_rectilinear_to_rectilinear(src_cube, tgt_grid)

    expected_data = np.empty([h, tgt_lats, t, tgt_lons, e])
    expected_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(height, 0)
    expected_cube.add_aux_coord(tgt_grid.coord("latitude"), (1, 3))
    expected_cube.add_dim_coord(time, 2)
    expected_cube.add_aux_coord(tgt_grid.coord("longitude"), (1, 3))
    expected_cube.add_aux_coord(extra, 4)
    expected_cube.add_aux_coord(spanning, [0, 2, 4])
    expected_cube = _add_metadata(expected_cube)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and coords.
    result.data = expected_data
    assert expected_cube == result


def test_extra_dims_curvilinear_to_rectilinear():
    """
    Test for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`.

    Tests the handling of extra dimensions and metadata. Ensures that proper
    coordinates, attributes, names and units are copied over.
    """
    h = 2
    t = 4
    e = 6
    src_lats = 3
    src_lons = 5

    tgt_lats = 5
    tgt_lons = 3

    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    src_grid = _curvilinear_cube(
        src_lons,
        src_lats,
        lon_bounds,
        lat_bounds,
    )
    tgt_grid = _grid_cube(
        tgt_lons,
        tgt_lats,
        lon_bounds,
        lat_bounds,
    )

    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")
    extra = AuxCoord(np.arange(e), long_name="extra dim")
    spanning = AuxCoord(np.ones([h, t, e]), long_name="spanning dim")

    src_data = np.empty([h, src_lats, t, src_lons, e])
    src_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    src_cube = Cube(src_data)
    src_cube.add_dim_coord(height, 0)
    src_cube.add_aux_coord(src_grid.coord("latitude"), (1, 3))
    src_cube.add_dim_coord(time, 2)
    src_cube.add_aux_coord(src_grid.coord("longitude"), (1, 3))
    src_cube.add_aux_coord(extra, 4)
    src_cube.add_aux_coord(spanning, [0, 2, 4])

    def _add_metadata(cube):
        result = cube.copy()
        result.units = "K"
        result.attributes = {"a": 1}
        result.standard_name = "air_temperature"
        scalar_height = AuxCoord([5], units="m", standard_name="height")
        scalar_time = DimCoord([10], units="s", standard_name="time")
        result.add_aux_coord(scalar_height)
        result.add_aux_coord(scalar_time)
        return result

    src_cube = _add_metadata(src_cube)

    result = regrid_rectilinear_to_rectilinear(src_cube, tgt_grid)

    expected_data = np.empty([h, tgt_lats, t, tgt_lons, e])
    expected_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(height, 0)
    expected_cube.add_dim_coord(tgt_grid.coord("latitude"), 1)
    expected_cube.add_dim_coord(time, 2)
    expected_cube.add_dim_coord(tgt_grid.coord("longitude"), 3)
    expected_cube.add_aux_coord(extra, 4)
    expected_cube.add_aux_coord(spanning, [0, 2, 4])
    expected_cube = _add_metadata(expected_cube)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and coords.
    result.data = expected_data
    assert expected_cube == result


def test_extra_dims_rectilinear_to_curvilinear():
    """
    Test for :func:`esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`.

    Tests the handling of extra dimensions and metadata. Ensures that proper
    coordinates, attributes, names and units are copied over.
    """
    h = 2
    t = 4
    e = 6
    src_lats = 3
    src_lons = 5

    tgt_lats = 5
    tgt_lons = 3

    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    src_grid = _grid_cube(
        src_lons,
        src_lats,
        lon_bounds,
        lat_bounds,
    )
    tgt_grid = _curvilinear_cube(
        tgt_lons,
        tgt_lats,
        lon_bounds,
        lat_bounds,
    )

    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")
    extra = AuxCoord(np.arange(e), long_name="extra dim")
    spanning = AuxCoord(np.ones([h, t, e]), long_name="spanning dim")

    src_data = np.empty([h, src_lats, t, src_lons, e])
    src_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    src_cube = Cube(src_data)
    src_cube.add_dim_coord(height, 0)
    src_cube.add_dim_coord(src_grid.coord("latitude"), 1)
    src_cube.add_dim_coord(time, 2)
    src_cube.add_dim_coord(src_grid.coord("longitude"), 3)
    src_cube.add_aux_coord(extra, 4)
    src_cube.add_aux_coord(spanning, [0, 2, 4])

    def _add_metadata(cube):
        result = cube.copy()
        result.units = "K"
        result.attributes = {"a": 1}
        result.standard_name = "air_temperature"
        scalar_height = AuxCoord([5], units="m", standard_name="height")
        scalar_time = DimCoord([10], units="s", standard_name="time")
        result.add_aux_coord(scalar_height)
        result.add_aux_coord(scalar_time)
        return result

    src_cube = _add_metadata(src_cube)

    result = regrid_rectilinear_to_rectilinear(src_cube, tgt_grid)

    expected_data = np.empty([h, tgt_lats, t, tgt_lons, e])
    expected_data[:] = np.arange(t * h * e).reshape([h, t, e])[
        :, np.newaxis, :, np.newaxis, :
    ]

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(height, 0)
    expected_cube.add_aux_coord(tgt_grid.coord("latitude"), (1, 3))
    expected_cube.add_dim_coord(time, 2)
    expected_cube.add_aux_coord(tgt_grid.coord("longitude"), (1, 3))
    expected_cube.add_aux_coord(extra, 4)
    expected_cube.add_aux_coord(spanning, [0, 2, 4])
    expected_cube = _add_metadata(expected_cube)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and coords.
    result.data = expected_data
    assert expected_cube == result
