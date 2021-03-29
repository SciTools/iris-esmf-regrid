"""Unit tests for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`."""

from iris.coords import AuxCoord, DimCoord
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import (
    regrid_unstructured_to_rectilinear,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__cube_to_GridInfo import (
    _grid_cube,
)
from esmf_regrid.tests.unit.experimental.unstructured_scheme.test__regrid_unstructured_to_rectilinear__prepare import (
    _flat_mesh_cube,
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
    result = regrid_unstructured_to_rectilinear(src, tgt)

    expected_data = np.ones([5, 6])
    expected_cube = _add_metadata(tgt)

    # Lenient check for data.
    assert np.allclose(expected_data, result.data)

    # Check metadata and scalar coords.
    expected_cube.data = result.data
    assert expected_cube == result
