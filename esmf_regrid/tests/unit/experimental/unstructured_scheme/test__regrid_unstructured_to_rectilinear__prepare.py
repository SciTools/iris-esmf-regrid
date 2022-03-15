"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme._regrid_unstructured_to_rectilinear__prepare`."""

from iris.coords import AuxCoord
from iris.cube import Cube
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo
from esmf_regrid.experimental.unstructured_regrid import MeshInfo
from esmf_regrid.experimental.unstructured_scheme import (
    _regrid_unstructured_to_rectilinear__prepare,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__cube_to_GridInfo import (
    _grid_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__mesh_to_MeshInfo import (
    _example_mesh,
)


def _full_mesh():
    mesh = _example_mesh()

    # In order to add a mesh to a cube, face locations must be added.
    # These are not used in calculations and are here given a value of zero.
    mesh_length = mesh.connectivity(contains_face=True).shape[0]
    dummy_face_lon = AuxCoord(np.zeros(mesh_length), standard_name="longitude")
    dummy_face_lat = AuxCoord(np.zeros(mesh_length), standard_name="latitude")
    mesh.add_coords(face_x=dummy_face_lon, face_y=dummy_face_lat)
    mesh.long_name = "example mesh"
    return mesh


def _flat_mesh_cube():
    """
    Return a 1D cube with a mesh attached.

    Returned cube has no metadata except for the mesh and two MeshCoords.
    Returned cube has data consisting of an array of ones.
    """
    mesh = _full_mesh()
    mesh_length = mesh.connectivity(contains_face=True).shape[0]

    cube = Cube(np.ones([mesh_length]))
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    cube.add_aux_coord(mesh_coord_x, 0)
    cube.add_aux_coord(mesh_coord_y, 0)
    return cube


def test_flat_cubes():
    """
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme._regrid_unstructured_to_rectilinear__prepare`.

    Tests with flat cubes as input (a 1D mesh cube and a 2D grid cube).
    """
    src = _flat_mesh_cube()

    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    regrid_info = _regrid_unstructured_to_rectilinear__prepare(
        src, tgt, method="conservative"
    )
    mesh_dim, grid_x, grid_y, regridder = regrid_info

    assert mesh_dim == 0
    assert grid_x == tgt.coord("longitude")
    assert grid_y == tgt.coord("latitude")
    assert type(regridder.tgt) == GridInfo
    assert type(regridder.src) == MeshInfo
