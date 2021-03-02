"""Unit tests for miscellaneous helper functions in `esmf_regrid.experimental.unstructured_scheme`."""

from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_scheme import _cube_to_GridInfo


def _grid_cube_global(n_lats, n_lons):
    lat_span = np.linspace(-90, 90, n_lats * 2 + 1)
    lat_points = lat_span[1::2]
    lat_bound_span = lat_span[::2]
    lat_bounds = np.stack([lat_bound_span[:-1], lat_bound_span[1:]], axis=-1)
    lat = DimCoord(lat_points, "latitude", units="degrees", bounds=lat_bounds)

    lon_span = np.linspace(-180, 180, n_lons * 2 + 1)
    lon_points = lon_span[1::2]
    lon_bound_span = lon_span[::2]
    lon_bounds = np.stack([lon_bound_span[:-1], lon_bound_span[1:]], axis=-1)
    lon = DimCoord(
        lon_points, "longitude", units="degrees", bounds=lon_bounds, circular=True
    )

    data = np.zeros([n_lats, n_lons])
    cube = Cube(data)
    cube.add_dim_coord(lat, 0)
    cube.add_dim_coord(lon, 1)
    return cube


def _grid_cube_local(n_lats, n_lons):
    lat_span = np.linspace(20, 60, n_lats * 2 + 1)
    lat_points = lat_span[1::2]
    lat_bound_span = lat_span[::2]
    lat_bounds = np.stack([lat_bound_span[:-1], lat_bound_span[1:]], axis=-1)
    lat = DimCoord(lat_points, "latitude", units="degrees", bounds=lat_bounds)

    lon_span = np.linspace(-20, 20, n_lons * 2 + 1)
    lon_points = lon_span[1::2]
    lon_bound_span = lon_span[::2]
    lon_bounds = np.stack([lon_bound_span[:-1], lon_bound_span[1:]], axis=-1)
    lon = DimCoord(
        lon_points, "longitude", units="degrees", bounds=lon_bounds, circular=False
    )

    data = np.zeros([n_lats, n_lons])
    cube = Cube(data)
    cube.add_dim_coord(lat, 0)
    cube.add_dim_coord(lon, 1)
    return cube


def test_cube_to_GridInfo():
    """
    Basic test for converting cubes to Gridinfo with
    :func:`~esmf_regrid.experimental.unstructured_scheme._cube_to_GridInfo`.
    """
    n_lats = 5
    n_lons = 6

    cube = _grid_cube_global(n_lats, n_lons)
    gridinfo = _cube_to_GridInfo(cube)
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    rg = Regridder(gridinfo, gridinfo)
    expected_weights = scipy.sparse.identity(n_lats*n_lons)
    # The following code provides efficient sparse array comparison.
    assert (expected_weights != rg.weight_matrix).nnz == 0

    cube = _grid_cube_local(n_lats, n_lons)
    gridinfo = _cube_to_GridInfo(cube)
    _ = gridinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # Note that this test fails when longitude is circular.
    rg = Regridder(gridinfo, gridinfo)
    assert (expected_weights != rg.weight_matrix).nnz == 0
