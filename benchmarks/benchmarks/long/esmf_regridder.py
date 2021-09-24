"""Slower benchmarks for :mod:`esmf_regrid.esmf_regridder`."""

import numpy as np
import dask.array as da
import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube

from benchmarks import disable_repeat_between_setup
from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube


class PrepareScalabilityGridToGrid:
    params = [50, 100, 500, 1000, 2000]
    height = 100
    regridder = ESMFAreaWeightedRegridder

    def src_cube(self, n):
        src = _grid_cube(n, n, lon_bounds, lat_bounds)
        return src

    def tgt_cube(self, n):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        grid = _grid_cube(n + 1, n + 1, lon_bounds, lat_bounds)
        return grid

    def setup(self, n):
        self.src = self.src_cube(n)
        self.tgt = self.tgt_cube(n)

    def time_prepare(self, n):
        rg = self.regridder(self.src, self.tgt)


class PrepareScalabilityMeshToGrid(PrepareScalabilityGridToGrid):
    regridder = MeshToGridESMFRegridder

    def src_cube(self, n):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        src = Cube(np.ones([n * n]))
        mesh = _gridlike_mesh(n, n)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        src.add_aux_coord(mesh_coord_x, 0)
        src.add_aux_coord(mesh_coord_y, 0)
        return src


class PrepareScalabilityGridToMesh(PrepareScalabilityGridToGrid):
    regridder = GridToMeshESMFRegridder

    def tgt_cube(self, n):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        tgt = Cube(np.ones([(n + 1) * (n + 1)]))
        mesh = _gridlike_mesh(n + 1, n + 1)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        return tgt


@disable_repeat_between_setup
class PerformScalabilityGridToGrid:
    params = [10, 100, 1000, 10000]
    grid_size = 500
    chunk_size = [grid_size / 2, grid_size / 2, 10]
    regridder = ESMFAreaWeightedRegridder

    def src_cube(self, height):
        data = da.ones([self.grid_size, self.grid_size, height], chunks=self.chunk_size)
        src = Cube(data)
        return src

    def add_src_metadata(self, cube):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        grid = _grid_cube(self.grid_size, self.grid_size, lon_bounds, lat_bounds)
        cube.add_dim_coord(grid.coord("latitude"), 0)
        cube.add_dim_coord(grid.coord("longitude"), 1)
        return cube

    def tgt_cube(self):
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        grid = _grid_cube(
            self.grid_size / 10 + 1, self.grid_size / 10 + 1, lon_bounds, lat_bounds
        )
        return grid

    def setup_cache(self, height):
        SYNTH_DATA_DIR = Path().cwd() / "tmp_data"
        SYNTH_DATA_DIR.mkdir(exist_ok=True)
        file = str(SYNTH_DATA_DIR.joinpath("chunked_cube.nc"))

        src = self.src_cube(height)
        iris.save(src, file, chunksizes=self.chunk_size)
        loaded_src = iris.load_cube(file)
        loaded_src = self.add_src_metadata(loaded_src)
        tgt = self.tgt_cube()
        rg = self.regridder(loaded_src, tgt)
        return rg, file

    def setup(self, cache, height):
        regridder, file = cache
        self.src = iris.load_cube(file)
        # Realise data.
        self.src.data
        cube = iris.load_cube(file)
        self.result = regridder(cube)

    def time_perform(self, cache, height):
        assert not self.src.has_lazy_data()
        rg, _ = cache
        _ = rg(self.src)

    def time_lazy_perform(self, cache, height):
        assert self.result.has_lazy_data()
        _ = self.result.data


class PerformScalabilityMeshToGrid(PerformScalabilityGridToGrid):
    regridder = MeshToGridESMFRegridder
    chunk_size = [grid_size / 2 * grid_size / 2, 10]

    def src_cube(self, height, chunk_size):
        data = da.ones(
            [self.grid_size * self.grid_size, height], chunks=self.chunk_size
        )
        src = Cube(data)
        return src

    def add_src_metadata(self, cube):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        mesh = _gridlike_mesh(grid_size, grid_size)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        cube.add_aux_coord(mesh_coord_x, 0)
        cube.add_aux_coord(mesh_coord_y, 0)
        return cube


class PerformScalabilityGridToMesh(PerformScalabilityGridToGrid):
    regridder = GridToMeshESMFRegridder
    chunk_size = [grid_size / 2 * grid_size / 2, 10]

    def tgt_cube(self):
        from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
            _gridlike_mesh,
        )

        tgt = Cube(np.ones([(self.grid_size / 10 + 1) * (self.grid_size / 10 + 1)]))
        mesh = _gridlike_mesh(self.grid_size / 10 + 1, self.grid_size / 10 + 1)
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
        tgt.add_aux_coord(mesh_coord_x, 0)
        tgt.add_aux_coord(mesh_coord_y, 0)
        return tgt
