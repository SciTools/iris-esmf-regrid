"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_unstructured`."""

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np
import pytest

from esmf_regrid.experimental.unstructured_scheme import (
    regrid_unstructured_to_unstructured,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
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
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_unstructured`.

    Tests with flat cubes as input (a 1D mesh cube and a 2D grid cube).
    """
    src = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    tgt = _flat_mesh_cube()
    # Ensure data in the target grid is different to the expected data.
    # i.e. target grid data is all zero, expected data is all one
    tgt.data[:] = 0

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    result = regrid_unstructured_to_unstructured(src, tgt)

    expected_data = np.ones([n_lats, n_lons])
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result


@pytest.mark.parametrize("method", ("bilinear", "nearest"))
def test_node_friendly_methods(method):
    """
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_unstructured`.

    Tests with the bilinear and nearest method.
    """
    n_lons = 6
    n_lats = 5
    src = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    tgt = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    # Ensure data in the target mesh is different to the expected data.
    # i.e. target mesh data is all zero, expected data is all one
    tgt.data[:] = 0

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    result = regrid_unstructured_to_unstructured(src, tgt, method=method)

    expected_data = np.ones_like(tgt.data)
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result


def test_invalid_args():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_unstructured`.

    Tests that an appropriate error is raised when arguments are invalid.
    """
    n_lons = 6
    n_lats = 5
    node_src = _gridlike_mesh_cube(n_lons, n_lats, location="node")
    edge_src = _gridlike_mesh_cube(n_lons, n_lats, location="edge")
    face_src = _gridlike_mesh_cube(n_lons, n_lats, location="face")
    tgt = _gridlike_mesh_cube(n_lons, n_lats)

    with pytest.raises(NotImplementedError):
        _ = regrid_unstructured_to_unstructured(face_src, tgt, method="other")
    with pytest.raises(ValueError) as excinfo:
        _ = regrid_unstructured_to_unstructured(node_src, tgt, method="conservative")
    expected_message = (
        "Conservative regridding requires a source cube located on "
        "the face of a cube, target cube had the node location."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = regrid_unstructured_to_unstructured(edge_src, tgt, method="bilinear")
    expected_message = (
        "bilinear regridding requires a source cube with a node "
        "or face location, target cube had the edge location."
    )
    assert expected_message in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _ = regrid_unstructured_to_unstructured(edge_src, tgt, method="nearest")
    expected_message = (
        "nearest regridding requires a source cube with a node "
        "or face location, target cube had the edge location."
    )
    assert expected_message in str(excinfo.value)


def test_multidim_cubes():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_unstructured`.

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
    cube = Cube(src_data)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    cube.add_aux_coord(mesh_coord_x, 1)
    cube.add_aux_coord(mesh_coord_y, 1)
    cube.add_dim_coord(time, 0)
    cube.add_dim_coord(height, 2)

    n_lons = 6
    n_lats = 5
    tgt = _gridlike_mesh_cube(n_lons, n_lats)

    result = regrid_unstructured_to_unstructured(cube, tgt)

    # Lenient check for data.
    expected_data = np.empty([t, n_lats * n_lons, h])
    expected_data[:] = np.arange(t * h).reshape(t, h)[:, np.newaxis, :]
    assert np.allclose(expected_data, result.data)

    expected_cube = Cube(expected_data)
    expected_cube.add_dim_coord(time, 0)
    expected_cube.add_aux_coord(tgt.coord("latitude"), 1)
    expected_cube.add_aux_coord(tgt.coord("longitude"), 1)
    expected_cube.add_dim_coord(height, 2)

    # Check metadata and scalar coords.
    result.data = expected_data
    assert expected_cube == result
