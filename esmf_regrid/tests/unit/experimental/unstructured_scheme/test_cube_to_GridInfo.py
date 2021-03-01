"""Unit tests for miscellaneous helper functions in `esmf_regrid.experimental.unstructured_scheme`."""

from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import _cube_to_GridInfo
from esmf_regrid.esmf_regridder import Regridder


def _grid_cube():
    n_lats = 50
    n_lons = 60

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


def test_cube_to_GridInfo():
    cube = _grid_cube()
    gridinfo = _cube_to_GridInfo(cube)
    _ = gridinfo.make_esmf_field()
    _ = Regridder(gridinfo, gridinfo)
