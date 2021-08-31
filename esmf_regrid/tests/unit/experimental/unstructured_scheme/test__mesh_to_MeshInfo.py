"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""

from iris.coords import AuxCoord
from iris.experimental.ugrid import Connectivity, Mesh
import numpy as np
from numpy import ma
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_scheme import _mesh_to_MeshInfo


def _pyramid_topology_connectivity_array():
    """
    Generate the face_node_connectivity array for a topological pyramid.

    The mesh described is a topological pyramid in the sense that there
    exists a polygonal base (described by the indices [0, 1, 2, 3, 4])
    and all other faces are triangles connected to a single node (the node
    with index 5).
    """
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
    # The base of the pyramid is the following pentagon.
    #
    # 60     0          3
    #        |  \    /  |
    # 10     |    4     |
    #        |          |
    #        |          |
    # -60    1----------2
    #
    #      120   180  -120
    #
    # The point of the pyramid is at the coordinate (0, 0).
    # The geometry is designed so that a valid ESMF object is only produced when
    # the orientation is correct (the face nodes are visited in an anticlockwise
    # order). This sensitivity is due to the base of the pyramid being convex.

    # Generate face_node_connectivity (fnc).
    fnc_ma = _pyramid_topology_connectivity_array()
    fnc = Connectivity(
        fnc_ma,
        cf_role="face_node_connectivity",
        start_index=0,
    )
    lon_values = [120, 120, -120, -120, 180, 0]
    lat_values = [60, -60, -60, 60, 10, 0]
    lons = AuxCoord(lon_values, standard_name="longitude")
    lats = AuxCoord(lat_values, standard_name="latitude")
    mesh = Mesh(2, ((lons, "x"), (lats, "y")), fnc)
    return mesh


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
    rg = Regridder(meshinfo, meshinfo)
    expected_weights = scipy.sparse.identity(meshinfo.size)
    assert np.allclose(expected_weights.todense(), rg.weight_matrix.todense())
