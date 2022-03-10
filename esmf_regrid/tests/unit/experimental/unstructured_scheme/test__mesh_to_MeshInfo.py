"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""

from iris.coords import AuxCoord
from iris.cube import Cube
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


def _gridlike_mesh(n_lons, n_lats):
    """
    Generate a global mesh with geometry similar to a rectilinear grid.

    The resulting mesh will have n_lons cells spanning its longitudes and
    n_lats cells spanning its latitudes for a total of (n_lons * n_lats) cells.
    Note that the cells neighbouring the poles will actually be triangular while
    the rest of the cells will be rectangular.
    """
    # Arrange the indices of the non-pole nodes in an array representative of their
    # latitude/longitude.
    fnc_template = np.arange((n_lats - 1) * n_lons).reshape(n_lats - 1, n_lons) + 1
    fnc_array = np.empty([n_lats, n_lons, 4])
    # Assign points in an anticlockwise orientation. From the 0 node to 1
    # longitude increases, then from 1 to 2 latitude increases, from 2 to 3
    # longitude decreases and from 3 to 0 latitude decreases.
    fnc_array[1:, :, 0] = fnc_template
    fnc_array[1:, :, 1] = np.roll(fnc_template, -1, 1)
    fnc_array[:-1, :, 2] = np.roll(fnc_template, -1, 1)
    fnc_array[:-1, :, 3] = fnc_template
    # Define the poles as single points. Note that all the cells adjacent to these
    # nodes will be topologically triangular with the pole node repeated. One of
    # these repeated pole node references will eventually be masked.
    fnc_array[0, :, :2] = 0
    num_nodes = fnc_template.max()
    fnc_array[-1, :, 2:] = num_nodes + 1
    # By convention, node references to be masked should be last in the list.
    # Since one of the pole node references will end up masked, this should be
    # moved to the end of the list of nodes.
    fnc_array[0, :, :] = np.roll(fnc_array[0, :, :], -1, -1)

    # One of the two references to the pole node are defined to be masked.
    fnc_mask = np.zeros_like(fnc_array)
    fnc_mask[0, :, -1] = 1
    fnc_mask[-1, :, -1] = 1
    fnc_ma = ma.array(fnc_array, mask=fnc_mask, dtype=int)

    # The face node connectivity is flattened to the correct dimensionality.
    fnc_ma = fnc_ma.reshape([-1, 4])

    # Describe the edge node connectivity.
    # There are n_lats * n_lons vertical edges and (n_lats - 1) * n_lons horizontal
    # edges which we arrange into an array for convenience of calculation.
    enc_array = np.empty([(n_lats * 2) - 1, n_lons, 2], dtype=int)
    # The vertical edges make up enc_array[:n_lats].
    enc_array[1:n_lats, :, 0] = fnc_template
    enc_array[: n_lats - 1, :, 1] = fnc_template
    enc_array[0, :, 0] = 0
    # The horizontal edges make up enc_array[n_lats:].
    enc_array[n_lats - 1, :, 1] = num_nodes + 1
    enc_array[n_lats:, :, 0] = fnc_template
    enc_array[n_lats:, :, 1] = np.roll(fnc_template, -1, 1)
    # The array is flattened to its proper shape of (N, 2).
    enc_array = enc_array.reshape([-1, 2])

    # Latitude and longitude values are set.
    lat_values = np.linspace(-90, 90, n_lats + 1)
    lon_values = np.linspace(-180, 180, n_lons, endpoint=False)
    # Latitude values are broadcast to arrays with the same shape as the face node
    # connectivity node references in fnc_template.
    lon_array, lat_array = np.meshgrid(lon_values, lat_values[1:-1])
    node_lats = np.empty(num_nodes + 2)
    # Note that fnc_template is created by reshaping a list of  indices. These
    # indices refer to node_lats and node_lons which are generated by reshaping
    # lat_array and lon_array. Because of the way reshaping preserves order, there
    # is a correspondance between an index in a particular position fnc_template
    # and the latitude and longitude described by lat_array and lon_array in the
    # same position.
    node_lats[1:-1] = lat_array.reshape([-1])
    # Define the latitude and longitude of the poles.
    node_lats[0] = lat_values[0]
    node_lats[-1] = lat_values[-1]
    node_lons = np.empty(num_nodes + 2)
    node_lons[1:-1] = lon_array.reshape([-1])
    node_lons[0] = 0
    node_lons[-1] = 0

    # Center Latitude and Longitude values are set.
    lon_centers = np.linspace(-180, 180, (2 * n_lons) + 1)[1::2]
    lat_centers = np.linspace(-90, 90, (2 * n_lats) + 1)[1::2]
    lon_center_array, lat_center_array = np.meshgrid(lon_centers, lat_centers)
    face_lons = lon_center_array.flatten()
    face_lats = lat_center_array.flatten()

    # Translate the mesh information into iris objects.
    fnc = Connectivity(
        fnc_ma,
        cf_role="face_node_connectivity",
        start_index=0,
    )
    enc = Connectivity(
        enc_array,
        cf_role="edge_node_connectivity",
        start_index=0,
    )
    lons = AuxCoord(node_lons, standard_name="longitude")
    lats = AuxCoord(node_lats, standard_name="latitude")
    mesh = Mesh(2, ((lons, "x"), (lats, "y")), [fnc, enc])

    # In order to add a mesh to a cube, face locations must be added.
    face_lon_coord = AuxCoord(face_lons, standard_name="longitude")
    face_lat_coord = AuxCoord(face_lats, standard_name="latitude")

    # Add dummy edge coords.
    dummy_points = np.zeros(enc_array.shape[0])
    edge_lon_coord = AuxCoord(dummy_points, standard_name="longitude")
    edge_lat_coord = AuxCoord(dummy_points, standard_name="latitude")

    mesh.add_coords(
        face_x=face_lon_coord,
        face_y=face_lat_coord,
        edge_x=edge_lon_coord,
        edge_y=edge_lat_coord,
    )
    mesh.long_name = "example mesh"
    return mesh


def _gridlike_mesh_cube(n_lons, n_lats, location="face"):
    mesh = _gridlike_mesh(n_lons, n_lats)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords(location)
    data = np.zeros_like(mesh_coord_x.points)
    cube = Cube(data)
    cube.add_aux_coord(mesh_coord_x, 0)
    cube.add_aux_coord(mesh_coord_y, 0)
    return cube


def test__mesh_to_MeshInfo():
    """Basic test for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""
    mesh = _example_mesh()
    meshinfo = _mesh_to_MeshInfo(mesh, location="face")

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
    meshinfo = _mesh_to_MeshInfo(mesh, location="face")

    # Ensure conversion to ESMF works without error.
    _ = meshinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    rg = Regridder(meshinfo, meshinfo)
    expected_weights = scipy.sparse.identity(meshinfo.size)
    assert np.allclose(expected_weights.todense(), rg.weight_matrix.todense())


def test_large_mesh_validity():
    """Test validity of objects derived from a large gridlike Mesh."""
    mesh = _gridlike_mesh(40, 20)
    meshinfo = _mesh_to_MeshInfo(mesh, location="face")

    # Ensure conversion to ESMF works without error.
    _ = meshinfo.make_esmf_field()

    # The following test ensures there are no overlapping cells.
    # This catches geometric/topological abnormalities that would arise from,
    # for example: switching lat/lon values, using euclidean coords vs spherical.
    rg = Regridder(meshinfo, meshinfo)
    expected_weights = scipy.sparse.identity(meshinfo.size)
    assert np.allclose(expected_weights.todense(), rg.weight_matrix.todense())
