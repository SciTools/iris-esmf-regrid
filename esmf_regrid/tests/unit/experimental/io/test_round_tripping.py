"""Unit tests for round tripping (saving then loading) with :mod:`esmf_regrid.experimental.io`."""

import numpy as np
from numpy import ma

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__cube_to_GridInfo import (
    _grid_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)


def _make_grid_to_mesh_regridder():
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
    tgt = _gridlike_mesh_cube(tgt_lons, tgt_lats)

    rg = GridToMeshESMFRegridder(src, tgt, mdtol=0.5)
    return rg, src


def _make_mesh_to_grid_regridder():
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
    src = _gridlike_mesh_cube(src_lons, src_lats)

    rg = MeshToGridESMFRegridder(src, tgt, mdtol=0.5)
    return rg, src


def test_GridToMeshESMFRegridder_round_trip(tmp_path):
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    original_rg, src = _make_grid_to_mesh_regridder()
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.mdtol == loaded_rg.mdtol
    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh

    # Compare the weight matrices.
    original_matrix = original_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # noqa E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Demonstrate regridding still gives the same results.
    src_data = np.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_mask = np.zeros(src.data.shape)
    src_mask[0, 0] = 1
    src.data = ma.array(src_data, mask=src_mask)
    # TODO: make this a cube comparison when mesh comparison becomes available.
    assert np.array_equal(original_rg(src).data, loaded_rg(src).data)

    # Ensure version data is equal.
    assert original_rg.regridder.esmf_version == loaded_rg.regridder.esmf_version
    assert (
        original_rg.regridder.esmf_regrid_version
        == loaded_rg.regridder.esmf_regrid_version
    )


def test_MeshToGridESMFRegridder_round_trip(tmp_path):
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    original_rg, src = _make_mesh_to_grid_regridder()
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.mdtol == loaded_rg.mdtol
    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh

    # Compare the weight matrices.
    original_matrix = original_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # noqa E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Demonstrate regridding still gives the same results.
    src_data = np.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_mask = np.zeros(src.data.shape)
    src_mask[0] = 1
    src.data = ma.array(src_data, mask=src_mask)
    assert original_rg(src) == loaded_rg(src)

    # Ensure version data is equal.
    assert original_rg.regridder.esmf_version == loaded_rg.regridder.esmf_version
    assert (
        original_rg.regridder.esmf_regrid_version
        == loaded_rg.regridder.esmf_regrid_version
    )
