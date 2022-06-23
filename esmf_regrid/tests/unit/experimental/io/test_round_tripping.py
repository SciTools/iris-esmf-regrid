"""Unit tests for round tripping (saving then loading) with :mod:`esmf_regrid.experimental.io`."""

import numpy as np
from numpy import ma

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)


def _make_grid_to_mesh_regridder(
    method="conservative", resolution=None, grid_dims=1, circular=True
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

    rg = GridToMeshESMFRegridder(
        src, tgt, method=method, mdtol=0.5, resolution=resolution
    )
    return rg, src


def _make_mesh_to_grid_regridder(
    method="conservative", resolution=None, grid_dims=1, circular=True
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

    rg = MeshToGridESMFRegridder(
        src, tgt, method=method, mdtol=0.5, resolution=resolution
    )
    return rg, src


def test_GridToMeshESMFRegridder_round_trip(tmp_path):
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    original_rg, src = _make_grid_to_mesh_regridder(circular=True)
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

    # Ensure resolution is equal.
    assert original_rg.resolution == loaded_rg.resolution
    original_res_rg, _ = _make_grid_to_mesh_regridder(resolution=8)
    res_filename = tmp_path / "regridder_res.nc"
    save_regridder(original_res_rg, res_filename)
    loaded_res_rg = load_regridder(str(res_filename))
    assert original_res_rg.resolution == loaded_res_rg.resolution
    assert (
        original_res_rg.regridder.src.resolution
        == loaded_res_rg.regridder.src.resolution
    )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, _ = _make_grid_to_mesh_regridder(circular=False)
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    assert original_nc_rg.grid_x == loaded_nc_rg.grid_x
    assert original_nc_rg.grid_y == loaded_nc_rg.grid_y


def test_GridToMeshESMFRegridder_bilinear_round_trip(tmp_path):
    """Test save/load round tripping for `GridToMeshESMFRegridder`."""
    original_rg, src = _make_grid_to_mesh_regridder(method="bilinear")
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


def test_MeshToGridESMFRegridder_round_trip(tmp_path):
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    original_rg, src = _make_mesh_to_grid_regridder(circular=True)
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

    # Ensure resolution is equal.
    assert original_rg.resolution == loaded_rg.resolution
    original_res_rg, _ = _make_mesh_to_grid_regridder(resolution=8)
    res_filename = tmp_path / "regridder_res.nc"
    save_regridder(original_res_rg, res_filename)
    loaded_res_rg = load_regridder(str(res_filename))
    assert original_res_rg.resolution == loaded_res_rg.resolution
    assert (
        original_res_rg.regridder.tgt.resolution
        == loaded_res_rg.regridder.tgt.resolution
    )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, _ = _make_grid_to_mesh_regridder(circular=False)
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    assert original_nc_rg.grid_x == loaded_nc_rg.grid_x
    assert original_nc_rg.grid_y == loaded_nc_rg.grid_y


def test_MeshToGridESMFRegridder_bilinear_round_trip(tmp_path):
    """Test save/load round tripping for `MeshToGridESMFRegridder`."""
    original_rg, src = _make_mesh_to_grid_regridder(method="bilinear")
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
