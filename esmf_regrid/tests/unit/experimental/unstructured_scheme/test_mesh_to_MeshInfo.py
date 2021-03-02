"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""

from iris.experimental.ugrid import Connectivity, Mesh
from iris.coords import AuxCoord
from numpy import ma
import numpy as np
import scipy.sparse

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_scheme import _mesh_to_MeshInfo


def _example_mesh():
    fnc_array = [
        [0, 1, 2, 3],
        [0, 1, 4, -1],
        [1, 2, 4, -1],
        [2, 3, 4, -1],
        [3, 0, 4, -1],
    ]
    fnc_mask = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    fnc_ma = ma.array(fnc_array, mask=fnc_mask)
    fnc = Connectivity(
        fnc_ma,
        cf_role="face_node_connectivity",
        start_index=0,
    )
    lon_values = [120, 120, -120, -120, 0]
    lat_values = [-60, 60, 60, -60, 0]
    lons = AuxCoord(lon_values, standard_name="longitude")
    lats = AuxCoord(lat_values, standard_name="latitude")
    mesh = Mesh(2, ((lons, "x"), (lats, "y")), fnc)
    return mesh


def test_mesh_info():
    """Basic test for :func:`esmf_regrid.experimental.unstructured_scheme._mesh_to_MeshInfo`."""
    mesh = _example_mesh()
    meshinfo = _mesh_to_MeshInfo(mesh)
    _ = meshinfo.make_esmf_field()

    rg = Regridder(meshinfo, meshinfo)
    expected_weights = scipy.sparse.identity(5)
    assert np.array_equal(expected_weights.todense(), rg.weight_matrix.todense())
