"""Quick running benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

from pathlib import Path

import numpy as np
import dask.array as da
import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube

from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
)


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
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    repeat = (5, 30, 20.0)
    # Prevent ASV running its warmup, which ignores `number` and would
    # therefore get a false idea of typical run time since the data would stop
    # being lazy.
    warmup_time = 0.0

    def setup_cache(self):
        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        # Rotated coord systems prevent pickling of the regridder so are
        # removed for the time being.
        # coord_system_src = RotatedGeogCS(0, 90, 90)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            # coord_system=coord_system_src,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        regridder = ESMFAreaWeightedRegridder(loaded_src, tgt)

        return regridder, file

    def setup(self, cache):
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data


class TimeMeshToGridRegridding:
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
        mesh = _gridlike_mesh(n_lons_src, n_lats_src)
        tgt = _grid_cube(
            n_lons_tgt,
            n_lats_tgt,
            lon_bounds,
            lat_bounds,
            coord_system=coord_system_src,
        )
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape([-1, h])
        src = Cube(src_data)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        src.add_aux_coord(mesh_coord_x, 0)
        src.add_aux_coord(mesh_coord_y, 0)
        self.regridder = MeshToGridESMFRegridder(src, tgt)
        self.src = src

    def time_perform_regridding(self, type):
        _ = self.regridder(self.src)


class TimeLazyMeshToGridRegridding:
    # Prevent repeat runs between setup() runs - data won't be lazy after 1st.
    number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    repeat = (5, 30, 20.0)
    # Prevent ASV running its warmup, which ignores `number` and would
    # therefore get a false idea of typical run time since the data would stop
    # being lazy.
    warmup_time = 0.0

    def setup_cache(self):
        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        mesh = _gridlike_mesh(n_lons_src, n_lats_src)
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src * n_lons_src, 10]
        src_data = da.ones([n_lats_src * n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        # While iris is not able to save meshes, we add these after loading.
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        loaded_src.add_aux_coord(mesh_coord_x, 0)
        loaded_src.add_aux_coord(mesh_coord_y, 0)
        regridder = MeshToGridESMFRegridder(loaded_src, tgt)

        return regridder, file, mesh

    def setup(self, cache):
        regridder, file, mesh = cache
        self.src = iris.load_cube(file)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        self.src.add_aux_coord(mesh_coord_x, 0)
        self.src.add_aux_coord(mesh_coord_y, 0)
        cube = iris.load_cube(file)
        cube.add_aux_coord(mesh_coord_x, 0)
        cube.add_aux_coord(mesh_coord_y, 0)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data


class TimeGridToMeshRegridding:
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
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt_data = np.zeros(n_lats_tgt * n_lons_tgt)
        tgt = Cube(tgt_data)
        mesh = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        self.regridder = GridToMeshESMFRegridder(src, tgt)
        self.src = src

    def time_perform_regridding(self, type):
        _ = self.regridder(self.src)


class TimeLazyGridToMeshRegridding:
    # Prevent repeat runs between setup() runs - data won't be lazy after 1st.
    number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    repeat = (5, 30, 20.0)
    # Prevent ASV running its warmup, which ignores `number` and would
    # therefore get a false idea of typical run time since the data would stop
    # being lazy.
    warmup_time = 0.0

    def setup_cache(self):
        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 100
        n_lats_src = 200
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 2000
        mesh = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        grid = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt_data = np.zeros(n_lats_tgt * n_lons_tgt)
        tgt = Cube(tgt_data)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        loaded_src = iris.load_cube(file)
        regridder = GridToMeshESMFRegridder(loaded_src, tgt)

        return regridder, file

    def setup(self, cache):
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        assert self.result.has_lazy_data()
        _ = self.result.data
