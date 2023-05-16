"""Unit tests for round tripping (saving then loading) with :mod:`esmf_regrid.experimental.io`."""

import numpy as np
from numpy import ma
import pytest

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)


def _make_grid_to_mesh_regridder(
    method="conservative",
    resolution=None,
    grid_dims=1,
    circular=True,
    masks=False,
):
    src_lons = 3
    src_lats = 4
    tgt_lons = 5
    tgt_lats = 6
    if circular:
        lon_bounds = (-180, 180)
    else:
        lon_bounds = (-180, 170)
    lat_bounds = (-90, 90)
    if grid_dims == 1:
        src = _grid_cube(src_lons, src_lats, lon_bounds, lat_bounds, circular=circular)
    else:
        src = _curvilinear_cube(src_lons, src_lats, lon_bounds, lat_bounds)
    src.coord("longitude").var_name = "longitude"
    src.coord("latitude").var_name = "latitude"
    if method == "bilinear":
        location = "node"
    else:
        location = "face"
    tgt = _gridlike_mesh_cube(tgt_lons, tgt_lats, location=location)

    if masks:
        src_data = ma.array(src.data)
        src_data[0, 0] = ma.masked
        src.data = src_data
        use_src_mask = True
        tgt_data = ma.array(tgt.data)
        tgt_data[0] = ma.masked
        tgt.data = tgt_data
        use_tgt_mask = True
    else:
        use_src_mask = False
        use_tgt_mask = False

    rg = GridToMeshESMFRegridder(
        src,
        tgt,
        method=method,
        mdtol=0.5,
        src_resolution=resolution,
        use_src_mask=use_src_mask,
        use_tgt_mask=use_tgt_mask,
    )
    return rg, src


def _make_mesh_to_grid_regridder(
    method="conservative", resolution=None, grid_dims=1, circular=True, masks=False
):
    src_lons = 3
    src_lats = 4
    tgt_lons = 5
    tgt_lats = 6
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    if grid_dims == 1:
        tgt = _grid_cube(tgt_lons, tgt_lats, lon_bounds, lat_bounds, circular=circular)
    else:
        tgt = _curvilinear_cube(tgt_lons, tgt_lats, lon_bounds, lat_bounds)
    tgt.coord("longitude").var_name = "longitude"
    tgt.coord("latitude").var_name = "latitude"
    if method == "bilinear":
        location = "node"
    else:
        location = "face"
    src = _gridlike_mesh_cube(src_lons, src_lats, location=location)

    if masks:
        src_data = ma.array(src.data)
        src_data[0] = ma.masked
        src.data = src_data
        use_src_mask = True
        tgt_data = ma.array(tgt.data)
        tgt_data[0, 0] = ma.masked
        tgt.data = tgt_data
        use_tgt_mask = True
    else:
        use_src_mask = False
        use_tgt_mask = False

    rg = MeshToGridESMFRegridder(
        src,
        tgt,
        method=method,
        mdtol=0.5,
        tgt_resolution=resolution,
        use_src_mask=use_src_mask,
        use_tgt_mask=use_tgt_mask,
    )
    return rg, src


@pytest.mark.parametrize("method", ["conservative", "bilinear", "nearest"])
def test_GridToMeshESMFRegridder_round_trip(tmp_path, method):
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    original_rg, src = _make_grid_to_mesh_regridder(method=method, circular=True)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.location == loaded_rg.location
    assert original_rg.method == loaded_rg.method
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
    src_data = ma.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_data[0, 0] = ma.masked
    src.data = src_data
    # TODO: make this a cube comparison when mesh comparison becomes available.
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)

    # Ensure version data is equal.
    assert original_rg.regridder.esmf_version == loaded_rg.regridder.esmf_version
    assert (
        original_rg.regridder.esmf_regrid_version
        == loaded_rg.regridder.esmf_regrid_version
    )

    if method == "conservative":
        # Ensure resolution is equal.
        assert original_rg.resolution == loaded_rg.resolution
        original_res_rg, _ = _make_grid_to_mesh_regridder(method=method, resolution=8)
        res_filename = tmp_path / "regridder_res.nc"
        save_regridder(original_res_rg, res_filename)
        loaded_res_rg = load_regridder(str(res_filename))
        assert original_res_rg.resolution == loaded_res_rg.resolution
        assert (
            original_res_rg.regridder.src.resolution
            == loaded_res_rg.regridder.src.resolution
        )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, _ = _make_grid_to_mesh_regridder(method=method, circular=False)
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    assert original_nc_rg.grid_x == loaded_nc_rg.grid_x
    assert original_nc_rg.grid_y == loaded_nc_rg.grid_y


def test_GridToMeshESMFRegridder_curvilinear_round_trip(tmp_path):
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    original_rg, src = _make_grid_to_mesh_regridder(grid_dims=2)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_data[0, 0] = ma.masked
    src.data = src_data
    # TODO: make this a cube comparison when mesh comparison becomes available.
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)


# TODO: parametrize the rest of the tests in this module.
@pytest.mark.parametrize(
    "rg_maker",
    [_make_grid_to_mesh_regridder, _make_mesh_to_grid_regridder],
    ids=["grid_to_mesh", "mesh_to_grid"],
)
def test_MeshESMFRegridder_masked_round_trip(tmp_path, rg_maker):
    """Test save/load round tripping for the Mesh regridder classes."""
    original_rg, src = rg_maker(masks=True)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    # Compare the weight matrices.
    original_matrix = original_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # noqa E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Ensure the masks are preserved
    assert np.array_equal(loaded_rg.src_mask, original_rg.src_mask)
    assert np.array_equal(loaded_rg.tgt_mask, original_rg.tgt_mask)


@pytest.mark.parametrize("method", ["conservative", "bilinear", "nearest"])
def test_MeshToGridESMFRegridder_round_trip(tmp_path, method):
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    original_rg, src = _make_mesh_to_grid_regridder(method=method, circular=True)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.location == loaded_rg.location
    assert original_rg.method == loaded_rg.method
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
    src_data = ma.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_data[0] = ma.masked
    src.data = src_data
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)

    # Ensure version data is equal.
    assert original_rg.regridder.esmf_version == loaded_rg.regridder.esmf_version
    assert (
        original_rg.regridder.esmf_regrid_version
        == loaded_rg.regridder.esmf_regrid_version
    )

    if method == "conservative":
        # Ensure resolution is equal.
        assert original_rg.resolution == loaded_rg.resolution
        original_res_rg, _ = _make_mesh_to_grid_regridder(method=method, resolution=8)
        res_filename = tmp_path / "regridder_res.nc"
        save_regridder(original_res_rg, res_filename)
        loaded_res_rg = load_regridder(str(res_filename))
        assert original_res_rg.resolution == loaded_res_rg.resolution
        assert (
            original_res_rg.regridder.tgt.resolution
            == loaded_res_rg.regridder.tgt.resolution
        )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, _ = _make_grid_to_mesh_regridder(method=method, circular=False)
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    assert original_nc_rg.grid_x == loaded_nc_rg.grid_x
    assert original_nc_rg.grid_y == loaded_nc_rg.grid_y


def test_MeshToGridESMFRegridder_curvilinear_round_trip(tmp_path):
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    original_rg, src = _make_mesh_to_grid_regridder(grid_dims=2)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert original_rg.grid_x == loaded_rg.grid_x
    assert original_rg.grid_y == loaded_rg.grid_y

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.product(src.data.shape)).reshape(src.data.shape)
    src_data[0] = ma.masked
    src.data = src_data
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)
