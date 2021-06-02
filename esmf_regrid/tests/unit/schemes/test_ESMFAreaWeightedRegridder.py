"""Unit tests for :func:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

import pytest

from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube


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
