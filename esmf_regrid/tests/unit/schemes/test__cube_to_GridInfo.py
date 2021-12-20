"""Unit tests for miscellaneous helper functions in `esmf_regrid.schemes`."""

from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.pp import EARTH_RADIUS
import numpy as np
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.schemes import _cube_to_GridInfo


def _generate_points_and_bounds(n, outer_bounds):
    lower, upper = outer_bounds
    full_span = np.linspace(lower, upper, n * 2 + 1)
    points = full_span[1::2]
    bound_span = full_span[::2]
    bounds = np.stack([bound_span[:-1], bound_span[1:]], axis=-1)
    return points, bounds


def _curvilinear_cube(
    n_lons,
    n_lats,
    lon_outer_bounds,
    lat_outer_bounds,
    coord_system=None,
):
    lon_points, lon_bounds = _generate_points_and_bounds(n_lons, lon_outer_bounds)
    lat_points, lat_bounds = _generate_points_and_bounds(n_lats, lat_outer_bounds)
    lon_bounds_full = np.empty([n_lats, n_lons, 4])
    lon_bounds_full[:, :, :2] = lon_bounds[np.newaxis, :]
    lon_bounds_full[:, :, 2:] = lon_bounds[np.newaxis, :, ::-1]
    lat_bounds_full = np.empty([n_lats, n_lons, 4])
    lat_bounds_full[:, :, :2] = lat_bounds[:, np.newaxis, 0, np.newaxis]
    lat_bounds_full[:, :, 2:] = lat_bounds[:, np.newaxis, 1, np.newaxis]

    lon_points, lat_points = np.meshgrid(lon_points, lat_points)
    lon = AuxCoord(
        lon_points,
        "longitude",
        units="degrees",
        bounds=lon_bounds_full,
        coord_system=coord_system,
    )
    lat = AuxCoord(
        lat_points,
        "latitude",
        units="degrees",
        bounds=lat_bounds_full,
        coord_system=coord_system,
    )

    data = np.zeros([n_lats, n_lons])
    cube = Cube(data)
    cube.add_aux_coord(lon, (0, 1))
    cube.add_aux_coord(lat, (0, 1))
    return cube


def _grid_cube(
    n_lons,
    n_lats,
    lon_outer_bounds,
    lat_outer_bounds,
    circular=False,
    coord_system=None,
):
    lon_points, lon_bounds = _generate_points_and_bounds(n_lons, lon_outer_bounds)
    lon = DimCoord(
        lon_points,
        "longitude",
        units="degrees",
        bounds=lon_bounds,
        circular=circular,
        coord_system=coord_system,
    )
    lat_points, lat_bounds = _generate_points_and_bounds(n_lats, lat_outer_bounds)
    lat = DimCoord(
        lat_points,
        "latitude",
        units="degrees",
        bounds=lat_bounds,
        coord_system=coord_system,
    )

    data = np.zeros([n_lats, n_lons])
    cube = Cube(data)
    cube.add_dim_coord(lon, 1)
    cube.add_dim_coord(lat, 0)
    return cube


def test_global_grid():
    """Test conversion of a global grid."""
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    cube = _grid_cube(
        n_lons,
        n_lats,
        lon_bounds,
        lat_bounds,
        circular=True,
        coord_system=GeogCS(EARTH_RADIUS),
    )
    gridinfo = _cube_to_GridInfo(cube)
    # Ensure conversion to ESMF works without error
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats * n_lons)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())
    assert gridinfo.crs == GeogCS(EARTH_RADIUS).as_cartopy_crs()


def test_local_grid():
    """Test conversion of a local grid."""
    n_lons = 6
    n_lats = 5
    lon_bounds = (-20, 20)
    lat_bounds = (20, 60)

    cube = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds)
    gridinfo = _cube_to_GridInfo(cube)
    # Ensure conversion to ESMF works without error
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # Note that this test fails when longitude is circular.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats * n_lons)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())


def test_grid_with_scalars():
    """Test conversion of a grid with scalar coords."""
    n_lons = 1
    n_lats = 5
    lon_bounds = (-20, 20)
    lat_bounds = (20, 60)

    cube = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds)
    # Convert longitude to a scalar
    cube = cube[:, 0]
    assert len(cube.shape) == 1

    gridinfo = _cube_to_GridInfo(cube)
    # Ensure conversion to ESMF works without error
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats * n_lons)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())


def test_curvilinear_grid():
    """Test conversion of a curvilinear global grid."""
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    cube = _curvilinear_cube(
        n_lons,
        n_lats,
        lon_bounds,
        lat_bounds,
        coord_system=GeogCS(EARTH_RADIUS),
    )
    gridinfo = _cube_to_GridInfo(cube)
    # Ensure conversion to ESMF works without error
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats * n_lons)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())
    assert gridinfo.crs == GeogCS(EARTH_RADIUS).as_cartopy_crs()

    # While curvilinear coords do not have the "circular" attribute, the code
    # allows "circular" to be True when setting the core regridder directly.
    # This describes an ESMF object which is topologically different, but ought
    # to be geometrically equivalent to the non-circular case.
    circular_gridinfo = _cube_to_GridInfo(cube)
    circular_gridinfo.circular = True
    rg_circular = Regridder(circular_gridinfo, gridinfo)
    assert np.allclose(expected_weights.todense(), rg_circular.weight_matrix.todense())
