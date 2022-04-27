"""Unit tests for miscellaneous helper functions in `esmf_regrid.experimental.unstructured_scheme`."""

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import _create_cube


def test_create_cube_2D():
    """Test creation of 2D output grid."""
    data = np.ones([2, 3])

    # Create a source cube with metadata and scalar coords
    src_cube = Cube(np.zeros(5))
    src_cube.units = "K"
    src_cube.attributes = {"a": 1}
    src_cube.standard_name = "air_temperature"
    scalar_height = AuxCoord([5], units="m", standard_name="height")
    scalar_time = DimCoord([10], units="s", standard_name="time")
    src_cube.add_aux_coord(scalar_height)
    src_cube.add_aux_coord(scalar_time)

    mesh_dim = 0

    grid_x = DimCoord(np.arange(3), standard_name="longitude")
    grid_y = DimCoord(np.arange(2), standard_name="latitude")

    cube = _create_cube(data, src_cube, (mesh_dim,), (grid_x, grid_y), 2)
    src_metadata = src_cube.metadata

    expected_cube = Cube(data)
    expected_cube.metadata = src_metadata
    expected_cube.add_dim_coord(grid_x, 1)
    expected_cube.add_dim_coord(grid_y, 0)
    expected_cube.add_aux_coord(scalar_height)
    expected_cube.add_aux_coord(scalar_time)
    assert expected_cube == cube


def test_create_cube_4D():
    """Test creation of 2D output grid."""
    data = np.ones([4, 2, 3, 5])

    # Create a source cube with metadata and scalar coords
    src_cube = Cube(np.zeros([4, 5, 5]))
    src_cube.units = "K"
    src_cube.attributes = {"a": 1}
    src_cube.standard_name = "air_temperature"
    scalar_height = AuxCoord([5], units="m", standard_name="height")
    scalar_time = DimCoord([10], units="s", standard_name="time")
    src_cube.add_aux_coord(scalar_height)
    src_cube.add_aux_coord(scalar_time)
    first_coord = DimCoord(np.arange(4), standard_name="air_pressure")
    src_cube.add_dim_coord(first_coord, 0)
    last_coord = AuxCoord(np.arange(5), long_name="last_coord")
    src_cube.add_aux_coord(last_coord, 2)
    multidim_coord = AuxCoord(np.ones([4, 5]), long_name="2d_coord")
    src_cube.add_aux_coord(multidim_coord, (0, 2))
    ignored_coord = AuxCoord(np.arange(5), long_name="ignore")
    src_cube.add_aux_coord(ignored_coord, 1)

    mesh_dim = 1

    grid_x = iris.coords.DimCoord(np.arange(3), standard_name="longitude")
    grid_y = iris.coords.DimCoord(np.arange(2), standard_name="latitude")

    cube = _create_cube(data, src_cube, (mesh_dim,), (grid_x, grid_y), 2)
    src_metadata = src_cube.metadata

    expected_cube = iris.cube.Cube(data)
    expected_cube.metadata = src_metadata
    expected_cube.add_dim_coord(grid_x, 2)
    expected_cube.add_dim_coord(grid_y, 1)
    expected_cube.add_dim_coord(first_coord, 0)
    expected_cube.add_aux_coord(last_coord, 3)
    expected_cube.add_aux_coord(multidim_coord, (0, 3))
    expected_cube.add_aux_coord(scalar_height)
    expected_cube.add_aux_coord(scalar_time)
    assert expected_cube == cube
