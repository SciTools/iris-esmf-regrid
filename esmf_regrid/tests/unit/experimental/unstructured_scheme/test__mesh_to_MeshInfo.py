"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""

from iris.coords import AuxCoord
from iris.experimental.ugrid import Connectivity, Mesh
import numpy as np
from numpy import ma
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_scheme import _mesh_to_MeshInfo


def _pyramid_topology_connectivity_array():
    """Generate the face_node_connectivity array for a topological pyramid."""
    fnc_array = [
        [0, 1, 2, 3, 4],
        [1, 0, 5, -1, -1],
        [2, 1, 5, -1, -1],
        [3, 2, 5, -1, -1],
        [4, 3, 5, -1, -1],
        [0, 4, 5, -1, -1],
    ]
    fnc_mask = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    fnc_ma = ma.array(fnc_array, mask=fnc_mask)
    return fnc_ma


def _example_mesh():
    """Generate a global mesh with a pentagonal pyramid topology."""
    # Generate face_node_connectivity (fnc).
    fnc_ma = _pyramid_topology_connectivity_array()
    fnc = Connectivity(
        fnc_ma,
        cf_role="face_node_connectivity",
        start_index=0,
    )
    # The geometry is designed so that this only produces a valid ESMF object
    # when the orientation is correct. This is due to the base of the pyramid
    # being convex.
    lon_values = [120, 120, -120, -120, 180, 0]
    lat_values = [60, -60, -60, 60, 10, 0]
    lons = AuxCoord(lon_values, standard_name="longitude")
    lats = AuxCoord(lat_values, standard_name="latitude")
    mesh = Mesh(2, ((lons, "x"), (lats, "y")), fnc)
    return mesh


def _check_equivalence(src_info, tgt_info):
    """
    Check that two objects describe the same physical space.

    This effectively checks that the ESMF mapping from src_info to tgt_info is identity.
    """
    assert src_info.size() == tgt_info.size()
    rg = Regridder(src_info, tgt_info)
    expected_weights = scipy.sparse.identity(src_info.size())
    assert np.allclose(expected_weights.todense(), rg.weight_matrix.todense())


def test__mesh_to_MeshInfo():
    """Basic test for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""
    mesh = _example_mesh()
    meshinfo = _mesh_to_MeshInfo(mesh)

    expected_nodes = np.array(
        [
            [120, 60],
            [120, -60],
            [-120, -60],
            [-120, 60],
            [180, 10],
            [0, 0],
        ]
    )
    assert np.array_equal(expected_nodes, meshinfo.node_coords)

    expected_connectivity = _pyramid_topology_connectivity_array()
    assert np.array_equal(expected_connectivity, meshinfo.fnc)

    expected_start_index = 0
    assert expected_start_index == meshinfo.esi


def test_anticlockwise_validity():
    """Test validity of objects derived from Mesh objects with anticlockwise orientation."""
    mesh = _example_mesh()
    meshinfo = _mesh_to_MeshInfo(mesh)

    # Ensure conversion to ESMF works without error.
    _ = meshinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    _check_equivalence(meshinfo, meshinfo)
