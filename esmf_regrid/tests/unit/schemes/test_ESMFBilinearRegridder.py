"""Unit tests for :class:`esmf_regrid.schemes.ESMFBilinearRegridder`."""

from cf_units import Unit
import numpy as np
import pytest

from esmf_regrid.schemes import ESMFBilinearRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)


def test_dim_switching():
    """
    Test calling of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    regridder = ESMFBilinearRegridder(src, tgt)
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
    Test calling of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    regridder = ESMFBilinearRegridder(src_init, tgt)

    msg = "The given cube is not defined on the same source grid as this regridder."
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_coord)
    with pytest.raises(ValueError, match=msg):
        _ = regridder(src_dif_circ)


def test_invalid_mdtol():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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
        _ = ESMFBilinearRegridder(src, tgt, mdtol=2)
    with pytest.raises(ValueError, match=match):
        _ = ESMFBilinearRegridder(src, tgt, mdtol=-1)


def test_curvilinear_equivalence():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    grid_to_grid = ESMFBilinearRegridder(grid_src, grid_tgt)
    grid_to_curv = ESMFBilinearRegridder(grid_src, curv_tgt)
    curv_to_grid = ESMFBilinearRegridder(curv_src, grid_tgt)
    curv_to_curv = ESMFBilinearRegridder(curv_src, curv_tgt)

    def extract_weights(regridder):
        return regridder.regridder.weight_matrix.todense()

    for regridder in [grid_to_curv, curv_to_grid, curv_to_curv]:
        assert np.allclose(extract_weights(grid_to_grid), extract_weights(regridder))


def test_curvilinear_and_rectilinear():
    """
    Test :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    rg = ESMFBilinearRegridder(src, tgt)
    result = rg(src)

    expected = grid_tgt.copy()
    # If the aux coords had been prioritised, expected.data would be a fully masked array.
    expected.data[:] = 1
    assert expected == result
    assert not np.ma.is_masked(result)


def test_unit_equivalence():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    grid_to_grid = ESMFBilinearRegridder(grid_src, grid_tgt)
    grid_rad_to_grid = ESMFBilinearRegridder(grid_src_rad, grid_tgt)
    grid_rad_to_curv = ESMFBilinearRegridder(grid_src_rad, curv_tgt)
    curv_to_grid_rad = ESMFBilinearRegridder(curv_src, grid_tgt_rad)
    curv_rad_to_grid = ESMFBilinearRegridder(curv_src_rad, grid_tgt)
    curv_to_curv_rad = ESMFBilinearRegridder(curv_src, curv_tgt_rad)

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
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

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

    rg_src_masked = ESMFBilinearRegridder(src_discontiguous, tgt, use_src_mask=True)
    rg_tgt_masked = ESMFBilinearRegridder(src, tgt_discontiguous, use_tgt_mask=True)
    rg_unmasked = ESMFBilinearRegridder(src, tgt)

    weights_src_masked = rg_src_masked.regridder.weight_matrix
    weights_tgt_masked = rg_tgt_masked.regridder.weight_matrix
    weights_unmasked = rg_unmasked.regridder.weight_matrix

    # Check there are no weights associated with the masked point.
    assert weights_src_masked[:, 0].nnz == 0
    assert weights_tgt_masked[0].nnz == 0

    # Check all other weights are correct.
    assert np.allclose(weights_src_masked[2:].todense(), weights_unmasked[2:].todense())
    assert np.allclose(weights_tgt_masked[2:].todense(), weights_unmasked[2:].todense())


def test_regrid_data():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinearRegridder`.

    Checks that regridding mathematics behaves in an expected way.
    """
    # Create two similar grids so that source data and expected result
    # data ought to look similar by visual inspection.
    src = _grid_cube(5, 4, [-180, 180], [-90, 90], circular=True)
    tgt = _grid_cube(4, 5, [-180, 180], [-90, 90], circular=True)

    src_data = np.arange(20).reshape([4, 5])
    src.data = src_data
    rg = ESMFBilinearRegridder(src, tgt)

    expected_data = np.array(
        [
            [
                0.3844461499074716,
                1.4148972061736933,
                2.585102793826307,
                3.615553850092528,
            ],
            [
                3.886915086072201,
                5.423003230276512,
                6.641000710423149,
                7.588216702776281,
            ],
            [
                7.649349191647961,
                8.891001259926682,
                10.108998740073318,
                11.35065080835204,
            ],
            [
                11.411783297223723,
                12.358999289576854,
                13.576996769723491,
                15.1130849139278,
            ],
            [
                15.384446149907474,
                16.414897206173688,
                17.58510279382631,
                18.615553850092528,
            ],
        ]
    )
    result = rg(src)
    np.testing.assert_allclose(expected_data, result.data)
