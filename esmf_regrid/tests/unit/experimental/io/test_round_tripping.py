"""Unit tests for round tripping (saving then loading) with :mod:`esmf_regrid.experimental.io`."""

from iris.cube import Cube
import numpy as np

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.tests import temp_filename
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__cube_to_GridInfo import (
    _grid_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
)


def test_GridToMeshESMFRegridder_round_trip():
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    src_lons = 3
    src_lats = 4
    tgt_lons = 5
    tgt_lats = 6
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    # TODO check that circularity is preserved.
    src = _grid_cube(src_lons, src_lats, lon_bounds, lat_bounds, circular=True)
    src.coord("longitude").var_name = "longitude"
    src.coord("latitude").var_name = "latitude"
    mesh = _gridlike_mesh(tgt_lons, tgt_lats)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    tgt_data = np.zeros(tgt_lons * tgt_lats)
    tgt = Cube(tgt_data)
    tgt.add_aux_coord(mesh_coord_x, 0)
    tgt.add_aux_coord(mesh_coord_y, 0)

    original_rg = GridToMeshESMFRegridder(src, tgt, mdtol=0.5)
    with temp_filename(suffix=".nc") as filename:
        save_regridder(original_rg, filename)
        loaded_rg = load_regridder(filename)
    assert original_rg.mdtol == loaded_rg.mdtol
    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh
    assert np.array_equal(
        original_rg.regridder.weight_matrix.todense(),
        loaded_rg.regridder.weight_matrix.todense(),
    )


def test_MeshToGridESMFRegridder_round_trip():
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    src_lons = 3
    src_lats = 4
    tgt_lons = 5
    tgt_lats = 6
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    # TODO check that circularity is preserved.
    tgt = _grid_cube(tgt_lons, tgt_lats, lon_bounds, lat_bounds, circular=True)
    tgt.coord("longitude").var_name = "longitude"
    tgt.coord("latitude").var_name = "latitude"
    mesh = _gridlike_mesh(src_lons, src_lats)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    src_data = np.zeros(src_lons * src_lats)
    src = Cube(src_data)
    src.add_aux_coord(mesh_coord_x, 0)
    src.add_aux_coord(mesh_coord_y, 0)

    original_rg = MeshToGridESMFRegridder(src, tgt, mdtol=0.5)
    with temp_filename(suffix=".nc") as filename:
        save_regridder(original_rg, filename)
        loaded_rg = load_regridder(filename)
    assert original_rg.mdtol == loaded_rg.mdtol
    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh
    assert np.array_equal(
        original_rg.regridder.weight_matrix.todense(),
        loaded_rg.regridder.weight_matrix.todense(),
    )
