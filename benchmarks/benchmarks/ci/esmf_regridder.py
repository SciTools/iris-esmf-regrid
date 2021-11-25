"""Quick running benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

import os
from pathlib import Path

import numpy as np
import dask.array as da
import iris
from iris.cube import Cube
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

from benchmarks import disable_repeat_between_setup
from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import ESMFAreaWeightedRegridder

from ..generate_data import _grid_cube, _gridlike_mesh_cube


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


class MultiGridCompare:
    params = ["similar", "large_source", "large_target", "mixed"]
    param_names = ["source/target difference"]

    def get_args(self, tp):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        n_lons_src = 20
        n_lats_src = 40
        n_lons_tgt = 20
        n_lats_tgt = 40
        h = 100
        if tp == "large_source":
            n_lons_src = 100
            n_lats_src = 200
        if tp == "large_target":
            n_lons_tgt = 100
            n_lats_tgt = 200
        alt_coord_system = tp == "mixed"
        args = (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            alt_coord_system,
        )
        return args


class TimeRegridding(MultiGridCompare):
    def setup(self, tp):
        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            alt_coord_system,
        ) = self.get_args(tp)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system,
        )
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        self.regrid_class = ESMFAreaWeightedRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt

    def time_prepare_regridding(self, tp):
        _ = self.regrid_class(self.src, self.tgt)

    def time_perform_regridding(self, tp):
        _ = self.regridder(self.src)


@disable_repeat_between_setup
class TimeLazyRegridding:
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
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            # alt_coord_system=True,
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


class TimeMeshToGridRegridding(TimeRegridding):
    def setup(self, tp):
        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            alt_coord_system_src,
        ) = self.get_args(tp)
        src_mesh = _gridlike_mesh_cube(n_lons_src, n_lats_src).mesh
        tgt = _grid_cube(
            n_lons_tgt,
            n_lats_tgt,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system_src,
        )
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape([-1, h])
        src = Cube(src_data)
        mesh_coord_x, mesh_coord_y = src_mesh.to_MeshCoords("face")
        src.add_aux_coord(mesh_coord_x, 0)
        src.add_aux_coord(mesh_coord_y, 0)

        self.regrid_class = MeshToGridESMFRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt


@disable_repeat_between_setup
class TimeLazyMeshToGridRegridding:
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
        src_mesh = _gridlike_mesh_cube(n_lons_src, n_lats_src).mesh
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src * n_lons_src, 10]
        src_data = da.ones([n_lats_src * n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        mesh_coord_x, mesh_coord_y = src_mesh.to_MeshCoords("face")
        src.add_aux_coord(mesh_coord_x, 0)
        src.add_aux_coord(mesh_coord_y, 0)

        iris.save(src, file, chunksizes=chunk_size)
        # Construct regridder with a loaded version of the grid for consistency.
        with PARSE_UGRID_ON_LOAD.context():
            loaded_src = iris.load_cube(file)
        regridder = MeshToGridESMFRegridder(loaded_src, tgt)

        return regridder, file

    def setup(self, cache):
        regridder, file = cache
        with PARSE_UGRID_ON_LOAD.context():
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


class TimeGridToMeshRegridding(TimeRegridding):
    def setup(self, tp):
        (
            lon_bounds,
            lat_bounds,
            n_lons_src,
            n_lats_src,
            n_lons_tgt,
            n_lats_tgt,
            h,
            alt_coord_system,
        ) = self.get_args(tp)
        grid = _grid_cube(
            n_lons_src,
            n_lats_src,
            lon_bounds,
            lat_bounds,
            alt_coord_system=alt_coord_system,
        )
        src_data = np.arange(n_lats_src * n_lons_src * h).reshape(
            [n_lats_src, n_lons_src, h]
        )
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
        self.regrid_class = GridToMeshESMFRegridder
        self.regridder = self.regrid_class(src, tgt)
        self.src = src
        self.tgt = tgt


@disable_repeat_between_setup
class TimeLazyGridToMeshRegridding:
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
        grid = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)

        chunk_size = [n_lats_src, n_lons_src, 10]
        src_data = da.ones([n_lats_src, n_lons_src, h], chunks=chunk_size)
        src = Cube(src_data)
        src.add_dim_coord(grid.coord("latitude"), 0)
        src.add_dim_coord(grid.coord("longitude"), 1)
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
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


class TimeRegridderIO(MultiGridCompare):
    def setup_cache(self):
        from esmf_regrid.experimental.io import save_regridder

        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)

        for tp in self.params:
            (
                lon_bounds,
                lat_bounds,
                n_lons_src,
                n_lats_src,
                n_lons_tgt,
                n_lats_tgt,
                _,
                alt_coord_system,
            ) = self.get_args(tp)
            src_grid = _grid_cube(
                n_lons_src,
                n_lats_src,
                lon_bounds,
                lat_bounds,
                alt_coord_system=alt_coord_system,
            )
            tgt_grid = _grid_cube(
                n_lons_tgt,
                n_lats_tgt,
                lon_bounds,
                lat_bounds,
                alt_coord_system=alt_coord_system,
            )
            src_mesh_cube = _gridlike_mesh_cube(
                n_lons_src,
                n_lats_src,
            )
            tgt_mesh_cube = _gridlike_mesh_cube(
                n_lons_tgt,
                n_lats_tgt,
            )

            mesh_to_grid_regridder = MeshToGridESMFRegridder(src_mesh_cube, tgt_grid)
            grid_to_mesh_regridder = GridToMeshESMFRegridder(src_grid, tgt_mesh_cube)

            source_file_m2g = str(SYNTH_DATA_DIR.joinpath(f"m2g_source_{tp}.nc"))
            source_file_g2m = str(SYNTH_DATA_DIR.joinpath(f"g2m_source_{tp}.nc"))

            save_regridder(mesh_to_grid_regridder, source_file_m2g)
            save_regridder(grid_to_mesh_regridder, source_file_g2m)
            return SYNTH_DATA_DIR

    def setup(self, SYNTH_DATA_DIR, tp):
        from esmf_regrid.experimental.io import load_regridder, save_regridder

        self.load_regridder = load_regridder
        self.save_regridder = save_regridder

        self.source_file_m2g = str(SYNTH_DATA_DIR.joinpath(f"m2g_source_{tp}.nc"))
        self.source_file_g2m = str(SYNTH_DATA_DIR.joinpath(f"g2m_source_{tp}.nc"))
        self.destination_file_m2g = str(
            SYNTH_DATA_DIR.joinpath(f"m2g_destination_{tp}.nc")
        )
        self.destination_file_g2m = str(
            SYNTH_DATA_DIR.joinpath(f"g2m_destination_{tp}.nc")
        )

        self.mesh_to_grid_regridder = load_regridder(self.source_file_m2g)
        self.grid_to_mesh_regridder = load_regridder(self.source_file_g2m)

    def teardown(self, _, tp):
        if os.path.exists(self.destination_file_m2g):
            os.remove(self.destination_file_m2g)
        if os.path.exists(self.destination_file_g2m):
            os.remove(self.destination_file_g2m)

    def time_save_mesh_to_grid(self, _, tp):
        self.save_regridder(self.mesh_to_grid_regridder, self.destination_file_m2g)

    def time_save_grid_to_mesh(self, _, tp):
        self.save_regridder(self.grid_to_mesh_regridder, self.destination_file_g2m)

    def time_load_mesh_to_grid(self, _, tp):
        _ = self.load_regridder(self.source_file_m2g)

    def time_load_grid_to_mesh(self, _, tp):
        _ = self.load_regridder(self.source_file_g2m)
