"""Integration test for regridding with different schedulers."""

import contextlib

import dask
import dask.array as da
import distributed
from iris.cube import Cube
import numpy as np
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _grid_cube,
)

from esmf_regrid.schemes import ESMFAreaWeighted


def _test_lazy_regridding():
    n_lons = 12
    n_lats = 10
    h = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    grid = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    src_data = np.arange(n_lats * n_lons * h, dtype=np.float32).reshape(
        [n_lats, n_lons, h]
    )
    src_data = da.from_array(src_data, chunks=[3, 5, 1])
    src = Cube(src_data)
    src.add_dim_coord(grid.coord("latitude"), 0)
    src.add_dim_coord(grid.coord("longitude"), 1)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    result = src.regrid(tgt, ESMFAreaWeighted())

    assert result.has_lazy_data()
    assert result.lazy_data().dtype == np.float64
    assert result.data.dtype == np.float64
    assert np.allclose(result.data, src_data)


@contextlib.contextmanager
def distributed_context():
    _distributed_client = distributed.Client()
    yield
    _distributed_client.close()


def test_distributed_scheduler():
    with distributed_context():
        _test_lazy_regridding()


def test_processes_scheduler():
    with dask.config.set(scheduler="processes"):
        _test_lazy_regridding()


def test_threads_scheduler():
    with dask.config.set(scheduler="threads"):
        _test_lazy_regridding()
