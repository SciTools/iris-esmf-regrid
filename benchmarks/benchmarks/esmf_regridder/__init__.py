"""Benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

import os
from pathlib import Path

import dask.array as da
import iris
from iris.cube import Cube
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from ..generate_data import _curvilinear_cube, _grid_cube, _gridlike_mesh_cube


def _make_small_grid_args():
    # Not importing the one in test_GridInfo - if that changes, these benchmarks
    #  would 'invisibly' change too.

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
    """Basic benchmarking for :class:~esmf_regrid.esmf_regridder.GridInfo`."""

    def setup(self):
        """ASV setup method."""
        lon, lat, lon_bounds, lat_bounds = _make_small_grid_args()
        self.grid = GridInfo(lon, lat, lon_bounds, lat_bounds)

    def time_make_grid(self):
        """Benchmark :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field` time."""
        esmf_grid = self.grid.make_esmf_field()
        esmf_grid.data[:] = 0

    time_make_grid.version = 1


class MultiGridCompare:
    """Mixin to prepare common arguments for benchmarking between different grid sizes."""

    params = ["similar", "large_source", "large_target", "mixed"]
    param_names = ["source/target difference"]

    def get_args(self, tp):
        """Prepare common arguments."""
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
    """Benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

    def setup(self, tp):
        """ASV setup method."""
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
        """Benchmark the prepare time."""
        _ = self.regrid_class(self.src, self.tgt)

    def time_perform_regridding(self, tp):
        """Benchmark the perform time."""
        _ = self.regridder(self.src)


class TimeLazyRegridding:
    """Lazy benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

    def setup_cache(self):
        """ASV setup_cache method."""
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
        """ASV setup method."""
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        """Benchmark the construction time of the lazy regridding operation."""
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        """Benchmark the final regridding operation time."""
        # Don't touch result.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.result.has_lazy_data()
        self.result.core_data().compute()


class TimeMeshToGridRegridding(TimeRegridding):
    """Benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.MeshToGridESMFRegridder`."""

    def setup(self, tp):
        """ASV setup method."""
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


class TimeLazyMeshToGridRegridding:
    """Lazy benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.MeshToGridESMFRegridder`."""

    def setup_cache(self):
        """ASV setup_cache method."""
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
        """ASV setup method."""
        regridder, file = cache
        with PARSE_UGRID_ON_LOAD.context():
            self.src = iris.load_cube(file)
            cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        """Benchmark the construction time of the lazy regridding operation."""
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        """Benchmark the final regridding operation time."""
        # Don't touch result.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.result.has_lazy_data()
        self.result.core_data().compute()


class TimeGridToMeshRegridding(TimeRegridding):
    """Benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.GridToMeshESMFRegridder`."""

    def setup(self, tp):
        """ASV setup method."""
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


class TimeLazyGridToMeshRegridding:
    """Lazy benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.GridToMeshESMFRegridder`."""

    def setup_cache(self):
        """ASV setup_cache method."""
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
        """ASV setup method."""
        regridder, file = cache
        self.src = iris.load_cube(file)
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_lazy_regridding(self, cache):
        """Benchmark the construction time of the lazy regridding operation."""
        assert self.src.has_lazy_data()
        regridder, _ = cache
        _ = regridder(self.src)

    def time_regridding_realisation(self, cache):
        """Benchmark the final regridding operation time."""
        # Don't touch result.data - permanent realisation plays badly with
        #  ASV's re-run strategy.
        assert self.result.has_lazy_data()
        self.result.core_data().compute()


class TimeRegridderIO(MultiGridCompare):
    """Benchmarks for regridder saving and loading."""

    params = [MultiGridCompare.params, ["mesh_to_grid", "grid_to_mesh"]]
    param_names = MultiGridCompare.param_names + ["regridder type"]

    def setup_cache(self):
        """ASV setup_cache method."""
        from esmf_regrid.experimental.io import save_regridder

        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)

        destination_file = str(SYNTH_DATA_DIR.joinpath("destination.nc"))
        file_dict = {"destination": destination_file}

        for tp in self.params[0]:
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

            rg_dict = {}
            rg_dict["mesh_to_grid"] = MeshToGridESMFRegridder(src_mesh_cube, tgt_grid)
            rg_dict["grid_to_mesh"] = GridToMeshESMFRegridder(src_grid, tgt_mesh_cube)

            for rgt in self.params[1]:
                regridder = rg_dict[rgt]
                source_file = str(SYNTH_DATA_DIR.joinpath(f"source_{tp}_{rgt}.nc"))

                save_regridder(regridder, source_file)

                file_dict[(tp, rgt)] = source_file
        return file_dict

    def setup(self, file_dict, tp, rgt):
        """ASV setup method."""
        from esmf_regrid.experimental.io import load_regridder, save_regridder

        self.load_regridder = load_regridder
        self.save_regridder = save_regridder

        self.source_file = file_dict[(tp, rgt)]
        self.destination_file = file_dict["destination"]
        self.regridder = load_regridder(self.source_file)

    def teardown(self, _, tp, rgt):
        """ASV teardown method."""
        if os.path.exists(self.destination_file):
            os.remove(self.destination_file)

    def time_save(self, _, tp, rgt):
        """Benchmark the saving time."""
        self.save_regridder(self.regridder, self.destination_file)

    def time_load(self, _, tp, rgt):
        """Benchmark the loading time."""
        _ = self.load_regridder(self.source_file)


class TimeMaskedRegridding:
    """Benchmarks for :class:`~esmf_regrid.esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

    def setup(self):
        """ASV setup method."""
        src = _curvilinear_cube(250, 251, [-180, 180], [-90, 90])
        tgt = _curvilinear_cube(251, 250, [-180, 180], [-90, 90])

        # Make src and tgt discontiguous at (0, 0)
        src_mask = np.zeros([250, 251], dtype=bool)
        src_mask[0, :] = True
        src.data = np.ma.array(src.data, mask=src_mask)
        src.coord("latitude").bounds[0, :, :2] = 0
        src.coord("longitude").bounds[0, :, :2] = 0

        tgt_mask = np.zeros([251, 250], dtype=bool)
        tgt_mask[:, 0] = True
        tgt.data = np.ma.array(tgt.data, mask=tgt_mask)
        tgt.coord("latitude").bounds[:, 0, ::3] = 0
        tgt.coord("longitude").bounds[:, 0, ::3] = 0

        self.regrid_class = ESMFAreaWeightedRegridder
        self.src = src
        self.tgt = tgt

    def time_prepare_without_masks(self):
        """Benchmark the prepare time with discontiguities but no mask."""
        _ = self.regrid_class(self.src, self.tgt)

    def time_prepare_with_masks(self):
        """Benchmark the prepare time with discontiguities and masks."""
        _ = self.regrid_class(self.src, self.tgt, src_mask=True, tgt_mask=True)
