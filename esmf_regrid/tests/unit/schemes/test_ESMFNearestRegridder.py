"""Unit tests for :class:`esmf_regrid.schemes.ESMFNearestRegridder`."""

from cf_units import Unit
import numpy as np
import pytest

from esmf_regrid.schemes import ESMFNearestRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)


def test_dim_switching():
    """
    Test calling of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

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

    regridder = ESMFNearestRegridder(src, tgt)
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
    Test calling of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

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

    regridder = ESMFNearestRegridder(src_init, tgt)

    msg = "The given cube is not defined on the same source grid as this regridder."
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_coord)
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_circ)


def test_curvilinear_equivalence():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

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

    grid_to_grid = ESMFNearestRegridder(grid_src, grid_tgt)
    grid_to_curv = ESMFNearestRegridder(grid_src, curv_tgt)
    curv_to_grid = ESMFNearestRegridder(curv_src, grid_tgt)
    curv_to_curv = ESMFNearestRegridder(curv_src, curv_tgt)

    def extract_weights(regridder):
        return regridder.regridder.weight_matrix.todense()

    for regridder in [grid_to_curv, curv_to_grid, curv_to_curv]:
        assert np.allclose(extract_weights(grid_to_grid), extract_weights(regridder))


def test_curvilinear_and_rectilinear():
    """
    Test :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

    Checks that a cube with both curvilinear and rectilinear coords still works.
    Checks that the DimCoords have priority over AuxCoords.
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

    src = curv_src.copy()
    grid_lat_src = grid_src.coord("latitude")
    grid_lat_src.standard_name = "grid_latitude"
    src.add_dim_coord(grid_lat_src, 0)
    grid_lon_src = grid_src.coord("longitude")
    grid_lon_src.standard_name = "grid_longitude"
    src.add_dim_coord(grid_lon_src, 1)

    tgt = curv_tgt.copy()
    grid_lat_tgt = grid_tgt.coord("latitude")
    grid_lat_tgt.standard_name = "grid_latitude"
    tgt.add_dim_coord(grid_lat_tgt, 0)
    grid_lon_tgt = grid_tgt.coord("longitude")
    grid_lon_tgt.standard_name = "grid_longitude"
    tgt.add_dim_coord(grid_lon_tgt, 1)

    # Change the AuxCoords to check that the DimCoords have priority.
    src.coord("latitude").bounds[:] = 0
    src.coord("longitude").bounds[:] = 0
    tgt.coord("latitude").bounds[:] = 0
    tgt.coord("longitude").bounds[:] = 0

    # Ensure the source data is all ones.
    src.data[:] = 1

    rg = ESMFNearestRegridder(src, tgt)
    result = rg(src)

    expected = grid_tgt.copy()
    # If the aux coords had been prioritised, expected.data would be a fully masked array.
    expected.data[:] = 1
    assert expected == result
    assert not np.ma.is_masked(result)


def test_unit_equivalence():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

    Checks that equivalent coordinates in degrees and radians give the same results.
    """
    # While this test has been copied from test_ESMFBilinerRegridder.py, a slight
    # change has been made to the parameter n_lons_src.
    # It should be noted that before this change this test was failing.
    # This is due to the fact that when two source points are equidistant from a
    # target point, rounding floating point differences due to unit conversion
    # would have an effect of the result.
    # These parameters have been tweaked so that no target point is equidistant from
    # two nearest source points.
    n_lons_src = 5
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

    grid_to_grid = ESMFNearestRegridder(grid_src, grid_tgt)
    grid_rad_to_grid = ESMFNearestRegridder(grid_src_rad, grid_tgt)
    grid_rad_to_curv = ESMFNearestRegridder(grid_src_rad, curv_tgt)
    curv_to_grid_rad = ESMFNearestRegridder(curv_src, grid_tgt_rad)
    curv_rad_to_grid = ESMFNearestRegridder(curv_src_rad, grid_tgt)
    curv_to_curv_rad = ESMFNearestRegridder(curv_src, curv_tgt_rad)

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


def test_masks():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

    Checks that the `use_src_mask` and `use_tgt_mask` keywords work properly.
    """
    src = _curvilinear_cube(7, 6, [-180, 180], [-90, 90])
    tgt = _curvilinear_cube(6, 7, [-180, 180], [-90, 90])

    # Make src and tgt discontiguous at (0, 0)
    src_mask = np.zeros([6, 7], dtype=bool)
    src_mask[0, 0] = True
    src.data = np.ma.array(src.data, mask=src_mask)
    src_discontiguous = src.copy()
    src_discontiguous.coord("latitude").bounds[0, 0] = 0
    src_discontiguous.coord("longitude").bounds[0, 0] = 0

    tgt_mask = np.zeros([7, 6], dtype=bool)
    tgt_mask[0, 0] = True
    tgt.data = np.ma.array(tgt.data, mask=tgt_mask)
    tgt_discontiguous = tgt.copy()
    tgt_discontiguous.coord("latitude").bounds[0, 0] = 0
    tgt_discontiguous.coord("longitude").bounds[0, 0] = 0

    rg_src_masked = ESMFNearestRegridder(src_discontiguous, tgt, use_src_mask=True)
    rg_tgt_masked = ESMFNearestRegridder(src, tgt_discontiguous, use_tgt_mask=True)
    rg_unmasked = ESMFNearestRegridder(src, tgt)

    weights_src_masked = rg_src_masked.regridder.weight_matrix
    weights_tgt_masked = rg_tgt_masked.regridder.weight_matrix
    weights_unmasked = rg_unmasked.regridder.weight_matrix

    # Check there are no weights associated with the masked point.
    assert weights_src_masked[:, 0].nnz == 0
    assert weights_tgt_masked[0].nnz == 0

    # Check other weights are correct. Note that unique to NEAREST_DTOS, masking a source
    # point causes the next nearest source point to gain weights in the weight matrix.
    # because of this, we ignore the row associated with that target point and check
    # the rest of the weights matrix.
    assert np.allclose(weights_src_masked[1:].todense(), weights_unmasked[1:].todense())
    assert np.allclose(weights_tgt_masked[1:].todense(), weights_unmasked[1:].todense())


@pytest.mark.parametrize(
    "src_type,tgt_type",
    [
        ("grid", "grid"),
        ("grid", "mesh"),
        ("mesh", "grid"),
        ("grid", "curv"),
        ("curv", "grid"),
    ],
)
def test_regrid_data(src_type, tgt_type):
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFNearestRegridder`.

    Checks that regridding mathematics behaves in an expected way.
    """
    # Create two similar grids so that source data and expected result
    # data ought to look similar by visual inspection.
    if src_type == "grid":
        src = _grid_cube(5, 4, [-180, 180], [-90, 90], circular=True)
    elif src_type == "mesh":
        src = _gridlike_mesh_cube(5, 4)
    elif src_type == "curv":
        src = _curvilinear_cube(5, 4, [-180, 180], [-90, 90])

    if tgt_type == "grid":
        tgt = _grid_cube(4, 5, [-180, 180], [-90, 90], circular=True)
    elif tgt_type == "mesh":
        tgt = _gridlike_mesh_cube(4, 5)
    elif tgt_type == "curv":
        tgt = _curvilinear_cube(4, 5, [-180, 180], [-90, 90])

    if src_type == "mesh":
        src.data = np.arange(20)
    else:
        src_data = np.arange(20).reshape([4, 5])
        src.data = src_data

    rg = ESMFNearestRegridder(src, tgt)

    # when two source points are equidistant from a
    # target point, the chosen source point is dependent on the index which ESMF
    # gives that point. This decision is described by ESMF to be arbitrary, but
    # ought to be consistent when dealing with the same precise grid.
    expected_data = np.array(
        [
            [0, 1, 3, 4],
            [5, 6, 8, 9],
            [5, 6, 8, 9],
            [10, 11, 13, 14],
            [15, 16, 18, 19],
        ]
    )
    if tgt_type == "mesh":
        expected_data = expected_data.flatten()

    result = rg(src)
    np.testing.assert_allclose(expected_data, result.data)
