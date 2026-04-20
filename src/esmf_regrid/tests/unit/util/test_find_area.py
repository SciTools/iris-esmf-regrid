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
    """Test find_area function on a mesh."""
    cube = _gridlike_mesh_cube(5, 6)
    areas = find_area(cube)
    assert areas.shape == (5 * 6,)
    # Check that the total area is equal to the area of the unit sphere, i.e. pi * 4
    assert np.allclose(areas.sum(), np.pi * 4)


def test_find_area_grid():
    """Test find_area function on a rectilinear grid."""
    # Generate a grid with regularly spaced latitudes and longitudes.
    cube = _grid_cube(5, 6, (-180, 180), (-90, 90), circular=True)
    areas = find_area(cube)
    assert areas.shape == (6, 5)
    # Check that cells between the same latitude bounds have approximately equal areas.
    assert np.allclose(areas[:, 1:], areas[:, :-1])
    # Check that equivalent latitude bounds above and below the equator are approximately equal.
    assert np.allclose(areas[:3], areas[:2:-1])
    # Check that cells are larger nearer the equator and smaller nearer the poles.
    assert np.all(areas[:2] < areas[1:3])
    # Check that the total area is equal to the area of the unit sphere, i.e. pi * 4
    assert np.allclose(areas.sum(), np.pi * 4)


def test_find_area_curvilinear_grid():
    """Test find_area function on a curvilinear grid."""
    cube = _curvilinear_cube(5, 6, (-180, 180), (-90, 90))
    areas = find_area(cube)
    assert areas.shape == (6, 5)
    # Check that the total area is equal to the area of the unit sphere, i.e. pi * 4
    assert np.allclose(areas.sum(), np.pi * 4)


def test_find_area_equivalence():
    """Test equivalence of find_area function on all grid/mesh types."""
    mesh_cube = _gridlike_mesh_cube(5, 6)
    grid_cube = _grid_cube(5, 6, (-180, 180), (-90, 90), circular=True)
    curvilinear_cube = _curvilinear_cube(5, 6, (-180, 180), (-90, 90))
    mesh_areas = find_area(mesh_cube)
    reshaped_mesh_areas = mesh_areas.reshape([6, 5])
    grid_areas = find_area(grid_cube)
    curvilinear_areas = find_area(curvilinear_cube)
    assert np.allclose(reshaped_mesh_areas, grid_areas)
    assert np.allclose(curvilinear_areas, grid_areas)


def test_find_area_radius():
    """Test radius keyword of find_area function."""
    cube = _gridlike_mesh_cube(5, 6)
    radius = 5000
    areas = find_area(cube, radius=radius)
    # Check that the total area is equal to the area of a sphere with the given radius.
    assert np.allclose(areas.sum(), np.pi * 4 * (radius**2))
