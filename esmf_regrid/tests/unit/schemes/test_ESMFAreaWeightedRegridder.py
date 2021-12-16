"""Unit tests for :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

from cf_units import Unit
import numpy as np
import pytest

from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)


def test_dim_switching():
    """
    Test calling of :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`.

    Checks that the regridder accepts a cube with dimensions in a different
    order than the cube which initialised it. Checks that dimension order is
    inherited from the cube in the calling function in both cases.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    regridder = ESMFAreaWeightedRegridder(src, tgt)
    unswitched_result = regridder(src)

    src_switched = src.copy()
    src_switched.transpose()
    switched_result = regridder(src_switched)

    assert unswitched_result.coord(dimensions=(0,)).standard_name == "latitude"
    assert unswitched_result.coord(dimensions=(1,)).standard_name == "longitude"
    assert switched_result.coord(dimensions=(0,)).standard_name == "longitude"
    assert switched_result.coord(dimensions=(1,)).standard_name == "latitude"


def test_differing_grids():
    """
    Test calling of :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`.

    Checks that the regridder raises an error when given a cube with a different
    grid to the one it was initialised with.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src_init = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    n_lons_dif = 7
    src_dif_coord = _grid_cube(
        n_lons_dif, n_lats, lon_bounds, lat_bounds, circular=True
    )
    src_dif_circ = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=False)

    regridder = ESMFAreaWeightedRegridder(src_init, tgt)

    msg = "The given cube is not defined on the same source grid as this regridder."
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_coord)
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_circ)


def test_invalid_mdtol():
    """
    Test initialisation of :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`.

    Checks that an error is raised when mdtol is out of range.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
    tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    match = "Value for mdtol must be in range 0 - 1, got "
    with pytest.raises(ValueError, match=match):
        _ = ESMFAreaWeightedRegridder(src, tgt, mdtol=2)
    with pytest.raises(ValueError, match=match):
        _ = ESMFAreaWeightedRegridder(src, tgt, mdtol=-1)


def test_curvilinear_equivalence():
    """
    Test initialisation of :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`.

    Checks that equivalent curvilinear and rectilinear coordinates give the same
    results.
    """
    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    grid_src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    grid_tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
    curv_src = _curvilinear_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)
    curv_tgt = _curvilinear_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

    grid_to_grid = ESMFAreaWeightedRegridder(grid_src, grid_tgt)
    grid_to_curv = ESMFAreaWeightedRegridder(grid_src, curv_tgt)
    curv_to_grid = ESMFAreaWeightedRegridder(curv_src, grid_tgt)
    curv_to_curv = ESMFAreaWeightedRegridder(curv_src, curv_tgt)

    def extract_weights(regridder):
        return regridder.regridder.weight_matrix.todense()

    for regridder in [grid_to_curv, curv_to_grid, curv_to_curv]:
        assert np.allclose(extract_weights(grid_to_grid), extract_weights(regridder))


def test_unit_equivalence():
    """
    Test initialisation of :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`.

    Checks that equivalent coordinates in degrees and radians give the same results.
    """
    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    lon_rad_bounds = (-np.pi, np.pi)
    lat_rad_bounds = (-np.pi / 2, np.pi / 2)

    def rad_coords(cube):
        cube.coord("latitude").units = Unit("radians")
        cube.coord("longitude").units = Unit("radians")

    grid_src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    grid_src_rad = _grid_cube(
        n_lons_src, n_lats_src, lon_rad_bounds, lat_rad_bounds, circular=True
    )
    rad_coords(grid_src_rad)
    grid_tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
    grid_tgt_rad = _grid_cube(
        n_lons_tgt, n_lats_tgt, lon_rad_bounds, lat_rad_bounds, circular=True
    )
    rad_coords(grid_tgt_rad)
    curv_src = _curvilinear_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)
    curv_src_rad = _curvilinear_cube(
        n_lons_src, n_lats_src, lon_rad_bounds, lat_rad_bounds
    )
    rad_coords(curv_src_rad)
    curv_tgt = _curvilinear_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
    curv_tgt_rad = _curvilinear_cube(
        n_lons_tgt, n_lats_tgt, lon_rad_bounds, lat_rad_bounds
    )
    rad_coords(curv_tgt_rad)

    grid_to_grid = ESMFAreaWeightedRegridder(grid_src, grid_tgt)
    grid_rad_to_grid = ESMFAreaWeightedRegridder(grid_src_rad, grid_tgt)
    grid_rad_to_curv = ESMFAreaWeightedRegridder(grid_src_rad, curv_tgt)
    curv_to_grid_rad = ESMFAreaWeightedRegridder(curv_src, grid_tgt_rad)
    curv_rad_to_grid = ESMFAreaWeightedRegridder(curv_src_rad, grid_tgt)
    curv_to_curv_rad = ESMFAreaWeightedRegridder(curv_src, curv_tgt_rad)

    def extract_weights(regridder):
        return regridder.regridder.weight_matrix.todense()

    for regridder in [
        grid_rad_to_grid,
        grid_rad_to_curv,
        curv_to_grid_rad,
        curv_rad_to_grid,
        curv_to_curv_rad,
    ]:
        assert np.allclose(extract_weights(grid_to_grid), extract_weights(regridder))
