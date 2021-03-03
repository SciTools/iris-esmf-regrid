"""Unit tests for miscellaneous helper functions in `esmf_regrid.experimental.unstructured_scheme`."""

from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_scheme import _cube_to_GridInfo


def _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=False):
    lon_lower, lon_upper = lon_bounds
    lon_span = np.linspace(lon_lower, lon_upper, n_lons * 2 + 1)
    lon_points = lon_span[1::2]
    lon_bound_span = lon_span[::2]
    lon_bounds = np.stack([lon_bound_span[:-1], lon_bound_span[1:]], axis=-1)
    lon = DimCoord(
        lon_points, "longitude", units="degrees", bounds=lon_bounds, circular=circular
    )

    lat_lower, lat_upper = lat_bounds
    lat_span = np.linspace(lat_lower, lat_upper, n_lats * 2 + 1)
    lat_points = lat_span[1::2]
    lat_bound_span = lat_span[::2]
    lat_bounds = np.stack([lat_bound_span[:-1], lat_bound_span[1:]], axis=-1)
    lat = DimCoord(lat_points, "latitude", units="degrees", bounds=lat_bounds)

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

    cube = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    gridinfo = _cube_to_GridInfo(cube)
    # Ensure conversion to ESMF works without error
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats * n_lons)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())


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
