"""Unit tests for miscellaneous helper functions in `esmf_regrid.experimental.unstructured_scheme`."""

import iris
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import _create_cube


def test_create_cube_2D():
    """Test creation of 2D output grid."""
    data = np.ones([2, 3])

    # Create a source cube with metadata and scalar coords
    src_cube = iris.cube.Cube(np.zeros(5))
    src_cube.units = "K"
    src_cube.attributes = {"a": 1}
    src_cube.standard_name = "air_temperature"
    scalar_height = iris.coords.AuxCoord([5], units="m", standard_name="height")
    scalar_time = iris.coords.DimCoord([10], units="s", standard_name="time")
    src_cube.add_aux_coord(scalar_height)
    src_cube.add_aux_coord(scalar_time)

    mesh_dim = 0
    # TODO: replace this with an actual mesh.
    mesh = None

    grid_x = iris.coords.DimCoord(np.arange(3), standard_name="longitude")
    grid_y = iris.coords.DimCoord(np.arange(2), standard_name="latitude")

    cube = _create_cube(data, src_cube, mesh_dim, mesh, grid_x, grid_y)
    src_metadata = src_cube.metadata
    src_scalars = src_cube.coords(dimensions=())
    assert src_metadata == cube.metadata
    assert src_scalars == cube.coords(dimensions=())
    assert cube.coord_dims(grid_x) == (1,)
    assert cube.coord_dims(grid_y) == (0,)
