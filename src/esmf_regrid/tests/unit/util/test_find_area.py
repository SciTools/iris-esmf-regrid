"""Unit tests for :mod:`esmf_regrid.util.find_area`."""

import numpy as np

from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)
from esmf_regrid.util import find_area


def test_find_area_mesh():
    cube = _gridlike_mesh_cube(5, 6)
    areas = find_area(cube)
    assert areas.shape == (5 * 6,)
    assert np.allclose(areas.sum(), np.pi * 4)


def test_find_area_grid():
    cube = _grid_cube(5, 6, (-180, 180), (-90, 90), circular=True)
    areas = find_area(cube)
    assert areas.shape == (6, 5)
    assert np.allclose(areas.sum(), np.pi * 4)


def test_find_area_curvilinear_grid():
    cube = _curvilinear_cube(5, 6, (-180, 180), (-90, 90))
    areas = find_area(cube)
    assert areas.shape == (6, 5)
    assert np.allclose(areas.sum(), np.pi * 4)
