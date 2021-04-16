"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`."""

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import (
    regrid_unstructured_to_rectilinear,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__cube_to_GridInfo import (
    _grid_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__regrid_unstructured_to_rectilinear__prepare import (
    _flat_mesh_cube,
    _full_mesh,
)


def test_flat_cubes():
    """
    Basic test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`.

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

    src = _add_metadata(src)
    src.data[:] = 1  # Ensure all data in the source is one.
    result = regrid_unstructured_to_rectilinear(src, tgt)

    expected_data = np.ones([n_lats, n_lons])
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result


def test_multidim_cubes():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`.

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
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    result = regrid_unstructured_to_rectilinear(cube, tgt)

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
