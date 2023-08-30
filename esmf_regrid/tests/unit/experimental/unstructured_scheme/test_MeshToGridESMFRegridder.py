"""Unit tests for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`."""

import dask.array as da
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np
import pytest

from esmf_regrid.experimental.unstructured_scheme import (
    MeshToGridESMFRegridder,
)
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
    _gridlike_mesh_cube,
)
from esmf_regrid.tests.unit.schemes.test__regrid_unstructured_to_rectilinear__prepare import (
    _flat_mesh_cube,
    _full_mesh,
)


def _add_metadata(cube):
    result = cube.copy()
    result.units = "K"
    result.attributes = {"a": 1}
    result.standard_name = "air_temperature"
    scalar_height = AuxCoord([5], units="m", standard_name="height")
    scalar_time = DimCoord([10], units="s", standard_name="time")
    result.add_aux_coord(scalar_height)
    result.add_aux_coord(scalar_time)
    return result


def test_flat_cubes():
    """
    Basic test for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Tests with flat cubes as input (a 1D mesh cube and a 2D grid cube).
    """
    src = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    # Ensure data in the target grid is different to the expected data.
    # i.e. target grid data is all zero, expected data is all one
    tgt.data[:] = 0

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    regridder = MeshToGridESMFRegridder(src, tgt)
    result = regridder(src)

    expected_data = np.ones([n_lats, n_lons])
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result


@pytest.mark.parametrize("nsi", [0, 1])
@pytest.mark.parametrize("method", ["bilinear", "nearest"])
def test_node_friendly_methods(method, nsi):
    """
    Basic test for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Tests with method="bilinear" and method="nearest".
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    face_src = _gridlike_mesh_cube(n_lons, n_lats, location="face", nsi=nsi)
    node_src = _gridlike_mesh_cube(n_lons, n_lats, location="node", nsi=nsi)

    face_src = _add_metadata(face_src)
    node_src = _add_metadata(node_src)
    # Ensure all data in the source is one.
    face_src.data[:] = 1
    node_src.data[:] = 1
    face_regridder = MeshToGridESMFRegridder(face_src, tgt, method=method)
    node_regridder = MeshToGridESMFRegridder(node_src, tgt, method=method)

    assert face_regridder.regridder.method == method
    assert node_regridder.regridder.method == method

    expected_data = np.ones_like(tgt.data)
    face_result = face_regridder(face_src)
    node_result = node_regridder(node_src)

    # Lenient check for data.
    assert np.allclose(expected_data, face_result.data)
    assert np.allclose(expected_data, node_result.data)

    # Check metadata and scalar coords.
    expected_cube = _add_metadata(tgt)
    expected_cube.data = face_result.data = node_result.data
    assert expected_cube == face_result == node_result


def test_multidim_cubes():
    """
    Test for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Tests with multidimensional cubes. The source cube contains
    coordinates on the dimensions before and after the mesh dimension.
    """
    mesh = _full_mesh()
    mesh_length = mesh.connectivity(contains_face=True).shape[0]

    h = 2
    t = 3
    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")

    src_data = np.empty([t, mesh_length, h])
    src_data[:] = np.arange(t * h).reshape([t, h])[:, np.newaxis, :]
    mesh_cube = Cube(src_data)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    mesh_cube.add_aux_coord(mesh_coord_x, 1)
    mesh_cube.add_aux_coord(mesh_coord_y, 1)
    mesh_cube.add_dim_coord(time, 0)
    mesh_cube.add_dim_coord(height, 2)

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    src_cube = mesh_cube.copy()
    src_cube.transpose([1, 0, 2])
    regridder = MeshToGridESMFRegridder(src_cube, tgt)
    result = regridder(mesh_cube)

    # Lenient check for data.
    expected_data = np.empty([t, n_lats, n_lons, h])
    expected_data[:] = np.arange(t * h).reshape(t, h)[:, np.newaxis, np.newaxis, :]
    assert np.allclose(expected_data, result.data)

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(time, 0)
    expected_cube.add_dim_coord(tgt.coord("latitude"), 1)
    expected_cube.add_dim_coord(tgt.coord("longitude"), 2)
    expected_cube.add_dim_coord(height, 3)

    # Check metadata and scalar coords.
    result.data = expected_data
    assert expected_cube == result


def test_invalid_mdtol():
    """
    Test initialisation of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that an error is raised when mdtol is out of range.
    """
    src = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(ValueError):
        _ = MeshToGridESMFRegridder(src, tgt, mdtol=2)
    with pytest.raises(ValueError):
        _ = MeshToGridESMFRegridder(src, tgt, mdtol=-1)


def test_invalid_method():
    """
    Test initialisation of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that an error is raised when method is invalid.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    face_src = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    edge_src = _gridlike_mesh_cube(n_lons, n_lats, location="edge")
    node_src = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(NotImplementedError):
        _ = MeshToGridESMFRegridder(face_src, tgt, method="other")
    with pytest.raises(ValueError) as excinfo:
        _ = MeshToGridESMFRegridder(node_src, tgt, method="conservative")
    expected_message = (
        "Conservative regridding requires a source cube located on "
        "the face of a cube, target cube had the node location."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = MeshToGridESMFRegridder(edge_src, tgt, method="bilinear")
    expected_message = (
        "bilinear regridding requires a source cube with a node "
        "or face location, target cube had the edge location."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = MeshToGridESMFRegridder(edge_src, tgt, method="nearest")
    expected_message = (
        "nearest regridding requires a source cube with a node "
        "or face location, target cube had the edge location."
    )
    assert expected_message in str(excinfo.value)


def test_invalid_resolution():
    """
    Test initialisation of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that an error is raised when the resolution is invalid.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(ValueError) as excinfo:
        _ = MeshToGridESMFRegridder(src, tgt, method="conservative", tgt_resolution=-1)
    expected_message = "resolution must be a positive integer."
    assert expected_message in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = MeshToGridESMFRegridder(src, tgt, method="bilinear", tgt_resolution=4)
    expected_message = "resolution can only be set for conservative regridding."
    assert expected_message in str(excinfo.value)


def test_default_mdtol():
    """
    Test initialisation of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that default mdtol values are as expected.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _gridlike_mesh_cube(n_lons, n_lats)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    rg_con = MeshToGridESMFRegridder(src, tgt, method="conservative")
    assert rg_con.mdtol == 1
    rg_bi = MeshToGridESMFRegridder(src, tgt, method="bilinear")
    assert rg_bi.mdtol == 0


@pytest.mark.xfail
def test_mistmatched_mesh():
    """
    Test the calling of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that an error is raised when the regridder is called with a cube
    whose mesh does not match the one used for initialisation.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    src = _gridlike_mesh_cube(n_lons, n_lats)
    other_loc = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    other_src = _flat_mesh_cube()

    rg = MeshToGridESMFRegridder(src, tgt)

    with pytest.raises(ValueError) as excinfo:
        _ = rg(tgt)
    expected_message = "The given cube is not defined on a mesh."
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = rg(other_loc)
    expected_message = (
        "The given cube is not defined on a the same "
        "mesh location as this regridder."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = rg(other_src)
    expected_message = (
        "The given cube is not defined on the same source mesh as this regridder."
    )
    assert expected_message in str(excinfo.value)


def test_laziness():
    """Test that regridding is lazy when source data is lazy."""
    n_lons = 12
    n_lats = 10
    h = 4
    i = 9
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    mesh = _gridlike_mesh(n_lons, n_lats)

    # Add a chunked dimension both before and after the mesh dimension.
    # The leading length 1 dimension matches the example in issue #135.
    src_data = np.arange(i * n_lats * n_lons * h).reshape([1, i, -1, h])
    src_data = da.from_array(src_data, chunks=[1, 3, 15, 2])
    src = Cube(src_data)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    src.add_aux_coord(mesh_coord_x, 2)
    src.add_aux_coord(mesh_coord_y, 2)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    rg = MeshToGridESMFRegridder(src, tgt)

    assert src.has_lazy_data()
    result = rg(src)
    assert result.has_lazy_data()
    out_chunks = result.lazy_data().chunks
    expected_chunks = ((1,), (3, 3, 3), (10,), (12,), (2, 2))
    assert out_chunks == expected_chunks
    assert np.allclose(result.data.reshape([1, i, -1, h]), src_data)


def test_resolution():
    """
    Test for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Tests for the tgt_resolution keyword.
    """
    mesh_cube = _flat_mesh_cube()

    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    lon_bands = _grid_cube(1, 4, lon_bounds, lat_bounds)
    lat_bands = _grid_cube(4, 1, lon_bounds, lat_bounds)

    resolution = 8

    lon_band_rg = MeshToGridESMFRegridder(
        mesh_cube, lon_bands, tgt_resolution=resolution
    )
    assert lon_band_rg.resolution == resolution
    assert lon_band_rg.regridder.tgt.resolution == resolution

    lat_band_rg = MeshToGridESMFRegridder(
        mesh_cube, lat_bands, tgt_resolution=resolution
    )
    assert lat_band_rg.resolution == resolution
    assert lat_band_rg.regridder.tgt.resolution == resolution


def test_curvilinear():
    """
    Test for :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Tests with curvilinear source cube.
    """
    mesh = _full_mesh()
    mesh_length = mesh.connectivity(contains_face=True).shape[0]

    h = 2
    t = 3
    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")

    src_data = np.empty([t, mesh_length, h])
    src_data[:] = np.arange(t * h).reshape([t, h])[:, np.newaxis, :]
    mesh_cube = Cube(src_data)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    mesh_cube.add_aux_coord(mesh_coord_x, 1)
    mesh_cube.add_aux_coord(mesh_coord_y, 1)
    mesh_cube.add_dim_coord(time, 0)
    mesh_cube.add_dim_coord(height, 2)

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _curvilinear_cube(n_lons, n_lats, lon_bounds, lat_bounds)

    src_cube = mesh_cube.copy()
    src_cube.transpose([1, 0, 2])
    regridder = MeshToGridESMFRegridder(src_cube, tgt)
    result = regridder(mesh_cube)
    mesh_cube_lazy = mesh_cube.copy(da.array(mesh_cube.data))
    result_lazy = regridder(mesh_cube_lazy)

    # Lenient check for data.
    expected_data = np.empty([t, n_lats, n_lons, h])
    expected_data[:] = np.arange(t * h).reshape(t, h)[:, np.newaxis, np.newaxis, :]
    assert np.allclose(expected_data, result.data)

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(time, 0)
    expected_cube.add_aux_coord(tgt.coord("latitude"), [1, 2])
    expected_cube.add_aux_coord(tgt.coord("longitude"), [1, 2])
    expected_cube.add_dim_coord(height, 3)

    # Check metadata and scalar coords.
    result.data = expected_data
    assert expected_cube == result
    assert result_lazy == result


@pytest.mark.parametrize(
    "resolution", (None, 2), ids=("no resolution", "with resolution")
)
def test_masks(resolution):
    """
    Test initialisation of :class:`esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Checks that the `use_src_mask` and `use_tgt_mask` keywords work properly.
    """
    src = _gridlike_mesh_cube(7, 6)
    if resolution is None:
        tgt = _curvilinear_cube(6, 7, [-180, 180], [-90, 90])
    else:
        # The resolution keyword is only valid for rectilinear grids.
        tgt = _grid_cube(6, 7, [-180, 180], [-90, 90])

    src_mask = np.zeros([6 * 7], dtype=bool)
    src_mask[0] = True
    src.data = np.ma.array(src.data, mask=src_mask)

    # Make tgt discontiguous at (0, 0)
    tgt_mask = np.zeros([7, 6], dtype=bool)
    tgt_mask[0, 0] = True
    tgt.data = np.ma.array(tgt.data, mask=tgt_mask)
    tgt_discontiguous = tgt.copy()
    if resolution is None:
        tgt_discontiguous.coord("latitude").bounds[0, 0] = 0
        tgt_discontiguous.coord("longitude").bounds[0, 0] = 0

    rg_src_masked = MeshToGridESMFRegridder(
        src, tgt, use_src_mask=True, tgt_resolution=resolution
    )
    rg_tgt_masked = MeshToGridESMFRegridder(
        src, tgt_discontiguous, use_tgt_mask=True, tgt_resolution=resolution
    )
    rg_unmasked = MeshToGridESMFRegridder(src, tgt, tgt_resolution=resolution)

    weights_src_masked = rg_src_masked.regridder.weight_matrix
    weights_tgt_masked = rg_tgt_masked.regridder.weight_matrix
    weights_unmasked = rg_unmasked.regridder.weight_matrix

    # Check there are no weights associated with the masked point.
    assert weights_src_masked[:, 0].nnz == 0
    assert weights_tgt_masked[0].nnz == 0

    # Check all other weights are correct.
    assert np.allclose(
        weights_src_masked[:, 1:].todense(), weights_unmasked[:, 1:].todense()
    )
    assert np.allclose(weights_tgt_masked[1:].todense(), weights_unmasked[1:].todense())
