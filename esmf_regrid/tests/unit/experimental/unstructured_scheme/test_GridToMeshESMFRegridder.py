"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`."""

import dask.array as da
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np
from numpy import ma
import pytest

from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
    _gridlike_mesh_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__regrid_unstructured_to_rectilinear__prepare import (
    _flat_mesh_cube,
)
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
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
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Tests with flat cubes as input (a 2D grid cube and a 1D mesh cube).
    """
    tgt = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    # Ensure data in the target grid is different to the expected data.
    # i.e. target grid data is all zero, expected data is all one
    tgt.data[:] = 0

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    regridder = GridToMeshESMFRegridder(src, tgt)
    result = regridder(src)
    src_T = src.copy()
    src_T.transpose()
    result_transposed = regridder(src_T)

    expected_data = np.ones([n_lats, n_lons])
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)
    assert np.allclose(expected_data, result_transposed.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result
    expected_cube.data = result_transposed.data
    assert expected_cube == result_transposed


def test_bilinear():
    """
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Tests with method="bilinear".
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    face_tgt = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    node_tgt = _gridlike_mesh_cube(n_lons, n_lats, location="node")

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    face_regridder = GridToMeshESMFRegridder(src, face_tgt, method="bilinear")
    node_regridder = GridToMeshESMFRegridder(src, node_tgt, method="bilinear")

    assert face_regridder.regridder.method == "bilinear"
    assert node_regridder.regridder.method == "bilinear"

    face_expected_data = np.ones_like(face_tgt.data)
    node_expected_data = np.ones_like(node_tgt.data)
    face_result = face_regridder(src)
    node_result = node_regridder(src)

    # Lenient check for data.
    assert np.allclose(face_expected_data, face_result.data)
    assert np.allclose(node_expected_data, node_result.data)

    # Check metadata and scalar coords.
    face_expected_cube = _add_metadata(face_tgt)
    node_expected_cube = _add_metadata(node_tgt)
    face_expected_cube.data = face_result.data
    node_expected_cube.data = node_result.data
    assert face_expected_cube == face_result
    assert node_expected_cube == node_result


def test_multidim_cubes():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_rectilinear_to_unstructured`.

    Tests with multidimensional cubes. The source cube contains
    coordinates on the dimensions before and after the grid dimensions.
    """
    tgt = _flat_mesh_cube()
    mesh = tgt.mesh
    mesh_length = mesh.connectivity(contains_face=True).shape[0]
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    grid = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    h = 2
    t = 3
    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")

    src_data = np.empty([t, n_lats, n_lons, h])
    src_data[:] = np.arange(t * h).reshape([t, h])[:, np.newaxis, np.newaxis, :]
    cube = Cube(src_data)
    cube.add_dim_coord(grid.coord("latitude"), 1)
    cube.add_dim_coord(grid.coord("longitude"), 2)
    cube.add_dim_coord(time, 0)
    cube.add_dim_coord(height, 3)

    regridder = GridToMeshESMFRegridder(grid, tgt)
    result = regridder(cube)

    # Lenient check for data.
    expected_data = np.empty([t, mesh_length, h])
    expected_data[:] = np.arange(t * h).reshape(t, h)[:, np.newaxis, :]
    assert np.allclose(expected_data, result.data)

    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(time, 0)
    expected_cube.add_aux_coord(mesh_coord_x, 1)
    expected_cube.add_aux_coord(mesh_coord_y, 1)
    expected_cube.add_dim_coord(height, 2)

    # Check metadata and scalar coords.
    result.data = expected_data
    assert expected_cube == result


def test_invalid_mdtol():
    """
    Test initialisation of :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Checks that an error is raised when mdtol is out of range.
    """
    tgt = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(ValueError):
        _ = GridToMeshESMFRegridder(src, tgt, mdtol=2)
    with pytest.raises(ValueError):
        _ = GridToMeshESMFRegridder(src, tgt, mdtol=-1)


def test_invalid_method():
    """
    Test initialisation of :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Checks that an error is raised when the method is invalid.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    face_tgt = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    edge_tgt = _gridlike_mesh_cube(n_lons, n_lats, location="edge")
    node_tgt = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(ValueError):
        _ = GridToMeshESMFRegridder(src, face_tgt, method="other")
    with pytest.raises(ValueError) as excinfo:
        _ = GridToMeshESMFRegridder(src, node_tgt, method="conservative")
    expected_message = (
        "Conservative regridding requires a target cube located on "
        "the face of a cube, target cube had the node location."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = GridToMeshESMFRegridder(src, edge_tgt, method="bilinear")
    expected_message = (
        "Bilinear regridding requires a target cube with a node "
        "or face location, target cube had the edge location."
    )
    assert expected_message in str(excinfo.value)


def test_invalid_resolution():
    """
    Test initialisation of :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Checks that an error is raised when the resolution is invalid.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(ValueError) as excinfo:
        _ = GridToMeshESMFRegridder(src, tgt, method="conservative", resolution=-1)
    expected_message = "resolution must be a positive integer."
    assert expected_message in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = GridToMeshESMFRegridder(src, tgt, method="bilinear", resolution=4)
    expected_message = "resolution can only be set for conservative regridding."
    assert expected_message in str(excinfo.value)


def test_default_mdtol():
    """
    Test initialisation of :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Checks that default mdtol values are as expected.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _gridlike_mesh_cube(n_lons, n_lats)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    rg_con = GridToMeshESMFRegridder(src, tgt, method="conservative")
    assert rg_con.mdtol == 1
    rg_bi = GridToMeshESMFRegridder(src, tgt, method="bilinear")
    assert rg_bi.mdtol == 0


def test_mismatched_grids():
    """
    Test error handling in calling of :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Checks that an error is raised when the regridder is called with a
    cube whose grid does not match with the one used when initialising
    the regridder.
    """
    tgt = _flat_mesh_cube()
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    regridder = GridToMeshESMFRegridder(src, tgt)

    n_lons_other = 3
    n_lats_other = 10
    src_other = _grid_cube(
        n_lons_other, n_lats_other, lon_bounds, lat_bounds, circular=True
    )
    with pytest.raises(ValueError):
        _ = regridder(src_other)


def test_mask_handling():
    """
    Test masked data handling for :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Tests masked data handling for multiple valid values for mdtol.
    """
    tgt = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    data = np.ones([n_lats, n_lons])
    mask = np.zeros([n_lats, n_lons])
    mask[0, 0] = 1
    masked_data = ma.array(data, mask=mask)
    src.data = masked_data
    regridder_0 = GridToMeshESMFRegridder(src, tgt, mdtol=0)
    regridder_05 = GridToMeshESMFRegridder(src, tgt, mdtol=0.05)
    regridder_1 = GridToMeshESMFRegridder(src, tgt, mdtol=1)
    result_0 = regridder_0(src)
    result_05 = regridder_05(src)
    result_1 = regridder_1(src)

    expected_data = np.ones(tgt.shape)
    expected_0 = ma.array(expected_data)
    expected_05 = ma.array(expected_data, mask=[0, 0, 1, 0, 0, 0])
    expected_1 = ma.array(expected_data, mask=[1, 0, 1, 0, 0, 0])

    assert ma.allclose(expected_0, result_0.data)
    assert ma.allclose(expected_05, result_05.data)
    assert ma.allclose(expected_1, result_1.data)


def test_laziness():
    """Test that regridding is lazy when source data is lazy."""
    n_lons = 12
    n_lats = 10
    h = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)

    mesh = _gridlike_mesh(n_lons, n_lats)

    src_data = np.arange(n_lats * n_lons * h).reshape([n_lats, n_lons, h])
    src_data = da.from_array(src_data, chunks=[3, 5, 2])
    src = Cube(src_data)
    grid = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    src.add_dim_coord(grid.coord("latitude"), 0)
    src.add_dim_coord(grid.coord("longitude"), 1)

    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    tgt_data = np.zeros([n_lats * n_lons])
    tgt = Cube(tgt_data)
    tgt.add_aux_coord(mesh_coord_x, 0)
    tgt.add_aux_coord(mesh_coord_y, 0)

    rg = GridToMeshESMFRegridder(src, tgt)

    assert src.has_lazy_data()
    result = rg(src)
    assert result.has_lazy_data()
    out_chunks = result.lazy_data().chunks
    expected_chunks = ((120,), (2, 2))
    assert out_chunks == expected_chunks
    assert np.allclose(result.data, src_data.reshape([-1, h]))


def test_resolution():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.

    Tests for the resolution keyword.
    """
    tgt = _flat_mesh_cube()
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    grid = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    resolution = 8

    result = GridToMeshESMFRegridder(grid, tgt, resolution=resolution)
    assert result.resolution == resolution
    assert result.regridder.src.resolution == resolution


def test_curvilinear():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_rectilinear_to_unstructured`.

    Tests with curvilinear target cube.
    """
    tgt = _flat_mesh_cube()
    mesh = tgt.mesh
    mesh_length = mesh.connectivity(contains_face=True).shape[0]
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    grid = _curvilinear_cube(n_lons, n_lats, lon_bounds, lat_bounds)

    h = 2
    t = 3
    height = DimCoord(np.arange(h), standard_name="height")
    time = DimCoord(np.arange(t), standard_name="time")

    src_data = np.empty([t, n_lats, n_lons, h])
    src_data[:] = np.arange(t * h).reshape([t, h])[:, np.newaxis, np.newaxis, :]
    cube = Cube(src_data)
    cube.add_aux_coord(grid.coord("latitude"), [1, 2])
    cube.add_aux_coord(grid.coord("longitude"), [1, 2])
    cube.add_dim_coord(time, 0)
    cube.add_dim_coord(height, 3)

    regridder = GridToMeshESMFRegridder(grid, tgt)
    result = regridder(cube)

    # Lenient check for data.
    expected_data = np.empty([t, mesh_length, h])
    expected_data[:] = np.arange(t * h).reshape(t, h)[:, np.newaxis, :]
    assert np.allclose(expected_data, result.data)

    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(time, 0)
    expected_cube.add_aux_coord(mesh_coord_x, 1)
    expected_cube.add_aux_coord(mesh_coord_y, 1)
    expected_cube.add_dim_coord(height, 2)

    # Check metadata and scalar coords.
    result.data = expected_data
    assert expected_cube == result
