"""Unit tests for round tripping (saving then loading) with :mod:`esmf_regrid.experimental.io`."""

import numpy as np
from numpy import ma
import pytest

from esmf_regrid import (
    Constants,
    ESMFAreaWeighted,
    ESMFAreaWeightedRegridder,
    ESMFBilinear,
    ESMFNearest,
)
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
    method=Constants.Method.CONSERVATIVE,
    regridder=GridToMeshESMFRegridder,
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
    if method == Constants.Method.BILINEAR:
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

    kwargs = {
        "mdtol": 0.5,
        "src_resolution": resolution,
        "use_src_mask": use_src_mask,
        "use_tgt_mask": use_tgt_mask,
    }
    if regridder == GridToMeshESMFRegridder:
        kwargs["method"] = method
    rg = regridder(src, tgt, **kwargs)
    return rg, src


def _make_mesh_to_grid_regridder(
    method=Constants.Method.CONSERVATIVE,
    regridder=MeshToGridESMFRegridder,
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
        tgt = _grid_cube(tgt_lons, tgt_lats, lon_bounds, lat_bounds, circular=circular)
    else:
        tgt = _curvilinear_cube(tgt_lons, tgt_lats, lon_bounds, lat_bounds)
    tgt.coord("longitude").var_name = "longitude"
    tgt.coord("latitude").var_name = "latitude"
    if method == Constants.Method.BILINEAR:
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

    kwargs = {
        "mdtol": 0.5,
        "tgt_resolution": resolution,
        "use_src_mask": use_src_mask,
        "use_tgt_mask": use_tgt_mask,
    }
    if regridder == MeshToGridESMFRegridder:
        kwargs["method"] = method
    rg = regridder(
        src,
        tgt,
        **kwargs,
    )
    return rg, src


def _compare_ignoring_var_names(x, y):
    old_var_name = x.var_name
    x.var_name = y.var_name
    assert x == y
    x.var_name = old_var_name


@pytest.mark.parametrize(
    "method,regridder",
    [
        (Constants.Method.CONSERVATIVE, GridToMeshESMFRegridder),
        (Constants.Method.BILINEAR, GridToMeshESMFRegridder),
        (Constants.Method.NEAREST, GridToMeshESMFRegridder),
        (None, ESMFAreaWeightedRegridder),
    ],
)
def test_grid_to_mesh_round_trip(tmp_path, method, regridder):
    """Test save/load round tripping for grid to mesh regridding."""
    original_rg, src = _make_grid_to_mesh_regridder(
        method=method, regridder=regridder, circular=True
    )
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    if regridder == GridToMeshESMFRegridder:
        assert original_rg.location == loaded_rg.location
        _compare_ignoring_var_names(original_rg.grid_x, loaded_rg.grid_x)
        _compare_ignoring_var_names(original_rg.grid_y, loaded_rg.grid_y)
    else:
        assert original_rg._tgt.location == loaded_rg._tgt.location
        _compare_ignoring_var_names(original_rg._src[0], loaded_rg._src[0])
        _compare_ignoring_var_names(original_rg._src[1], loaded_rg._src[1])
    assert original_rg.method == loaded_rg.method
    assert original_rg.mdtol == loaded_rg.mdtol
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh

    # Compare the weight matrices.
    original_matrix = original_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # noqa E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.prod(src.data.shape)).reshape(src.data.shape)
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

    if method == Constants.Method.CONSERVATIVE:
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
    elif regridder == ESMFAreaWeightedRegridder:
        assert original_rg.src_resolution == loaded_rg.src_resolution
        original_res_rg, _ = _make_grid_to_mesh_regridder(
            regridder=regridder, resolution=8
        )
        res_filename = tmp_path / "regridder_res.nc"
        save_regridder(original_res_rg, res_filename)
        loaded_res_rg = load_regridder(str(res_filename))
        assert original_res_rg.src_resolution == loaded_res_rg.src_resolution
        assert (
            original_res_rg.regridder.src.resolution
            == loaded_res_rg.regridder.src.resolution
        )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, src = _make_grid_to_mesh_regridder(
        method=method, regridder=regridder, circular=True
    )
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    if regridder == GridToMeshESMFRegridder:
        _compare_ignoring_var_names(original_nc_rg.grid_x, loaded_nc_rg.grid_x)
        _compare_ignoring_var_names(original_nc_rg.grid_y, loaded_nc_rg.grid_y)
    else:
        _compare_ignoring_var_names(original_nc_rg._src[0], loaded_nc_rg._src[0])
        _compare_ignoring_var_names(original_nc_rg._src[1], loaded_nc_rg._src[1])


@pytest.mark.parametrize(
    "regridder",
    [GridToMeshESMFRegridder, ESMFAreaWeightedRegridder],
)
def test_grid_to_mesh_curvilinear_round_trip(tmp_path, regridder):
    """Test save/load round tripping for grid to mesh regridding."""
    original_rg, src = _make_grid_to_mesh_regridder(regridder=regridder, grid_dims=2)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    if regridder == GridToMeshESMFRegridder:
        _compare_ignoring_var_names(original_rg.grid_x, loaded_rg.grid_x)
        _compare_ignoring_var_names(original_rg.grid_y, loaded_rg.grid_y)
    else:
        _compare_ignoring_var_names(original_rg._src[0], loaded_rg._src[0])
        _compare_ignoring_var_names(original_rg._src[1], loaded_rg._src[1])

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.prod(src.data.shape)).reshape(src.data.shape)
    src_data[0, 0] = ma.masked
    src.data = src_data
    # TODO: make this a cube comparison when mesh comparison becomes available.
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)


# TODO: parametrize the rest of the tests in this module.
@pytest.mark.parametrize(
    "regridder",
    ["unstructured", ESMFAreaWeightedRegridder],
)
@pytest.mark.parametrize(
    "rg_maker",
    [_make_grid_to_mesh_regridder, _make_mesh_to_grid_regridder],
    ids=["grid_to_mesh", "mesh_to_grid"],
)
def test_MeshESMFRegridder_masked_round_trip(tmp_path, rg_maker, regridder):
    """Test save/load round tripping for the Mesh regridder classes."""
    if regridder == "unstructured":
        original_rg, src = rg_maker(masks=True)
    else:
        original_rg, src = rg_maker(regridder=regridder, masks=True)
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


@pytest.mark.parametrize(
    "method,regridder",
    [
        (Constants.Method.CONSERVATIVE, MeshToGridESMFRegridder),
        (Constants.Method.BILINEAR, MeshToGridESMFRegridder),
        (Constants.Method.NEAREST, MeshToGridESMFRegridder),
        (None, ESMFAreaWeightedRegridder),
    ],
)
def test_mesh_to_grid_round_trip(tmp_path, method, regridder):
    """Test save/load round tripping for mesh to grid regridding."""
    original_rg, src = _make_mesh_to_grid_regridder(
        method=method, regridder=regridder, circular=True
    )
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    if regridder == MeshToGridESMFRegridder:
        assert original_rg.location == loaded_rg.location
        _compare_ignoring_var_names(original_rg.grid_x, loaded_rg.grid_x)
        _compare_ignoring_var_names(original_rg.grid_y, loaded_rg.grid_y)
    else:
        assert original_rg._src.location == loaded_rg._src.location
        _compare_ignoring_var_names(original_rg._tgt[0], loaded_rg._tgt[0])
        _compare_ignoring_var_names(original_rg._tgt[1], loaded_rg._tgt[1])

    assert original_rg.method == loaded_rg.method
    assert original_rg.mdtol == loaded_rg.mdtol
    # TODO: uncomment when iris mesh comparison becomes available.
    # assert original_rg.mesh == loaded_rg.mesh

    # Compare the weight matrices.
    original_matrix = original_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # noqa E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.prod(src.data.shape)).reshape(src.data.shape)
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

    if method == Constants.Method.CONSERVATIVE:
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
    elif regridder == ESMFAreaWeightedRegridder:
        assert original_rg.src_resolution == loaded_rg.src_resolution
        original_res_rg, _ = _make_mesh_to_grid_regridder(
            regridder=regridder, resolution=8
        )
        res_filename = tmp_path / "regridder_res.nc"
        save_regridder(original_res_rg, res_filename)
        loaded_res_rg = load_regridder(str(res_filename))
        assert original_res_rg.tgt_resolution == loaded_res_rg.tgt_resolution
        assert (
            original_res_rg.regridder.tgt.resolution
            == loaded_res_rg.regridder.tgt.resolution
        )

    # Ensure grid equality for non-circular coords.
    original_nc_rg, _ = _make_mesh_to_grid_regridder(
        method=method, regridder=regridder, circular=False
    )
    nc_filename = tmp_path / "non_circular_regridder.nc"
    save_regridder(original_nc_rg, nc_filename)
    loaded_nc_rg = load_regridder(str(nc_filename))
    if regridder == MeshToGridESMFRegridder:
        _compare_ignoring_var_names(original_nc_rg.grid_x, loaded_nc_rg.grid_x)
        _compare_ignoring_var_names(original_nc_rg.grid_y, loaded_nc_rg.grid_y)
    else:
        _compare_ignoring_var_names(original_nc_rg._tgt[0], loaded_nc_rg._tgt[0])
        _compare_ignoring_var_names(original_nc_rg._tgt[1], loaded_nc_rg._tgt[1])


@pytest.mark.parametrize(
    "regridder",
    [MeshToGridESMFRegridder, ESMFAreaWeightedRegridder],
)
def test_mesh_to_grid_curvilinear_round_trip(tmp_path, regridder):
    """Test save/load round tripping for mesh to grid regridding."""
    original_rg, src = _make_mesh_to_grid_regridder(regridder=regridder, grid_dims=2)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    if regridder == MeshToGridESMFRegridder:
        _compare_ignoring_var_names(original_rg.grid_x, loaded_rg.grid_x)
        _compare_ignoring_var_names(original_rg.grid_y, loaded_rg.grid_y)
    else:
        _compare_ignoring_var_names(original_rg._tgt[0], loaded_rg._tgt[0])
        _compare_ignoring_var_names(original_rg._tgt[1], loaded_rg._tgt[1])

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.prod(src.data.shape)).reshape(src.data.shape)
    src_data[0] = ma.masked
    src.data = src_data
    original_result = original_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)


@pytest.mark.parametrize(
    "src_type,tgt_type",
    [
        ("grid", "grid"),
        ("grid", "mesh"),
        ("mesh", "grid"),
        ("mesh", "mesh"),
    ],
)
@pytest.mark.parametrize(
    "scheme",
    [ESMFAreaWeighted, ESMFBilinear, ESMFNearest],
    ids=["conservative", "linear", "nearest"],
)
def test_generic_regridder(tmp_path, src_type, tgt_type, scheme):
    """Test save/load round tripping for regridders in `esmf_regrid.schemes`."""
    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    if src_type == "grid":
        src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    elif src_type == "mesh":
        src = _gridlike_mesh_cube(n_lons_src, n_lats_src)
    if tgt_type == "grid":
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
    elif tgt_type == "mesh":
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)

    original_rg = scheme().regridder(src, tgt)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    if src_type == "grid":
        assert original_rg._src == loaded_rg._src
    if tgt_type == "grid":
        assert original_rg._tgt == loaded_rg._tgt
    if scheme == ESMFAreaWeighted:
        assert original_rg.src_resolution == loaded_rg.src_resolution
        assert original_rg.tgt_resolution == loaded_rg.tgt_resolution
    assert original_rg.mdtol == loaded_rg.mdtol


@pytest.mark.parametrize(
    "src_type,tgt_type",
    [
        ("grid", "grid"),
        ("grid", "mesh"),
        ("mesh", "grid"),
        ("mesh", "mesh"),
    ],
)
@pytest.mark.parametrize(
    "scheme",
    [ESMFAreaWeighted, ESMFBilinear, ESMFNearest],
    ids=["conservative", "linear", "nearest"],
)
def test_generic_regridder_masked(tmp_path, src_type, tgt_type, scheme):
    """Test save/load round tripping for regridders in `esmf_regrid.schemes`."""
    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    if src_type == "grid":
        src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
        src.data = ma.array(src.data)
        src.data[0, 0] = ma.masked
    elif src_type == "mesh":
        src = _gridlike_mesh_cube(n_lons_src, n_lats_src)
        src.data = ma.array(src.data)
        src.data[0] = ma.masked
    if tgt_type == "grid":
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
        tgt.data = ma.array(tgt.data)
        tgt.data[0, 0] = ma.masked
    elif tgt_type == "mesh":
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
        tgt.data = ma.array(tgt.data)
        tgt.data[0] = ma.masked

    original_rg = scheme().regridder(src, tgt, use_src_mask=True, use_tgt_mask=True)
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert np.allclose(original_rg.src_mask, loaded_rg.src_mask)
    assert np.allclose(original_rg.tgt_mask, loaded_rg.tgt_mask)


@pytest.mark.parametrize(
    "scheme",
    [ESMFAreaWeighted],
    ids=["conservative"],
)
def test_generic_regridder_resolution(tmp_path, scheme):
    """Test save/load round tripping for regridders in `esmf_regrid.schemes`."""
    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
    src_resolution = 3
    tgt_resolution = 4

    original_rg = scheme().regridder(
        src, tgt, src_resolution=src_resolution, tgt_resolution=tgt_resolution
    )
    filename = tmp_path / "regridder.nc"
    save_regridder(original_rg, filename)
    loaded_rg = load_regridder(str(filename))

    assert loaded_rg.src_resolution == src_resolution
    assert loaded_rg.regridder.src.resolution == src_resolution
    assert loaded_rg.tgt_resolution == tgt_resolution
    assert loaded_rg.regridder.tgt.resolution == tgt_resolution
