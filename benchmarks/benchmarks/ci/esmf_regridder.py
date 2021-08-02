"""Quick running benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

from pathlib import Path
from shutil import rmtree

import numpy as np
import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube

from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube


SYNTH_DATA_DIR = Path().cwd() / "tmp_data"


def setup_cache(*args):
    SYNTH_DATA_DIR.mkdir(exist_ok=True)


def teardown(*args):
    rmtree(SYNTH_DATA_DIR)


def _make_small_grid_args():
    """
    Not importing the one in test_GridInfo - if that changes, these benchmarks
    would 'invisibly' change too.

    """
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


class TimeGridInfo:
    def setup(self):
        lon, lat, lon_bounds, lat_bounds = _make_small_grid_args()
        self.grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    def time_make_grid(self):
        """Basic test for :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
        esmf_grid = self.grid.make_esmf_field()
        esmf_grid.data[:] = 0

    time_make_grid.version = 1


class TimeRegridding:
    params = ["similar", "large source", "large target", "mixed"]
    param_names = ["source/target difference"]

    def setup(self, type):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 20
        n_lats_src = 40
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 100
        if type == "large source":
            n_lons_src = 100
            n_lats_src = 200
        if type == "large target":
            n_lons_tgt = 100
            n_lats_tgt = 200
        if type == "mixed":
            coord_system_src = RotatedGeogCS(0, 90, 90)
        else:
            coord_system_src = None
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            coord_system=coord_system_src,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        self.regridder = ESMFAreaWeightedRegridder(src, tgt)
        self.src = src

    def time_perform_regridding(self, type):
        _ = self.regridder(self.src)


class TimeLazyRegridding:
    # Prevent repeat runs between setup() runs - data won't be lazy after 1st.
    number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    repeat = (10, 10, 10.0)
    # Prevent ASV running its warmup, which ignores `number` and would
    # therefore get a false idea of typical run time since the data would stop
    # being lazy.
    warmup_time = 0.0

    file = SYNTH_DATA_DIR.joinpath("chunked_cube.nc")

    def setup_cache(self):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 100
        coord_system_src = RotatedGeogCS(0, 90, 90)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            coord_system=coord_system_src,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
        regridder = ESMFAreaWeightedRegridder(grid, tgt)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        print(5)
        iris.save(src, self.file, chunksizes=chunk_size)
        print(6)

        return regridder

    def setup(self, cache):
        self.src = iris.load_cube(self.file)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder = cache
        _ = regridder(self.src)
