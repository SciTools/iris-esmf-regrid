"""Unit tests for `esmf_regrid.schemes`."""

from iris.coord_systems import OSGB
import numpy as np
from numpy import ma
import pytest

from esmf_regrid.schemes import ESMFAreaWeighted, ESMFBilinear, ESMFNearest
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh,
    _gridlike_mesh_cube,
)


def _test_cube_regrid(scheme, src_type, tgt_type):
    """
    Test that `scheme` can be passed to a cubes regrid method.

    Checks that regridding occurs and that mdtol is used correctly.
    """
    if tgt_type == "just_mesh":
        scheme_default = scheme(tgt_location="face")
        scheme_full_mdtol = scheme(mdtol=1, tgt_location="face")
    else:
        scheme_default = scheme()
        scheme_full_mdtol = scheme(mdtol=1)

    n_lons_src = 6
    n_lons_tgt = 3
    n_lats_src = 4
    n_lats_tgt = 2
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    if src_type == "grid":
        src = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
        src_data = np.zeros([n_lats_src, n_lons_src])
        src_mask = np.zeros([n_lats_src, n_lons_src])
        src_mask[0, 0] = 1
    else:
        src = _gridlike_mesh_cube(n_lons_src, n_lats_src)
        src_data = np.zeros([n_lats_src * n_lons_src])
        src_mask = np.zeros([n_lats_src * n_lons_src])
        src_mask[0] = 1
    if tgt_type == "grid":
        tgt = _grid_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True)
        expected_data_default = np.zeros([n_lats_tgt, n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt, n_lons_tgt])
        expected_mask[0, 0] = 1
    elif tgt_type == "mesh":
        tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
        expected_data_default = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask[0] = 1
    elif tgt_type == "just_mesh":
        tgt = _gridlike_mesh(n_lons_tgt, n_lats_tgt)
        expected_data_default = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask = np.zeros([n_lats_tgt * n_lons_tgt])
        expected_mask[0] = 1

    src_data = ma.array(src_data, mask=src_mask)
    src.data = src_data

    result_default = src.regrid(tgt, scheme_default)
    result_full = src.regrid(tgt, scheme_full_mdtol)

    expected_data_full = ma.array(expected_data_default, mask=expected_mask)

    if tgt_type == "just_mesh":
        tgt_template = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
    else:
        tgt_template = tgt
    expected_cube_default = tgt_template.copy()
    expected_cube_default.data = expected_data_default

    expected_cube_full = tgt_template.copy()
    expected_cube_full.data = expected_data_full

    assert expected_cube_default == result_default
    assert expected_cube_full == result_full


def _test_invalid_mdtol(scheme):
    """
    Test initialisation of the scheme.

    Checks that an error is raised when mdtol is out of range.
    """
    match = "Value for mdtol must be in range 0 - 1, got "
    with pytest.raises(ValueError, match=match):
        _ = scheme(mdtol=2)
    with pytest.raises(ValueError, match=match):
        _ = scheme(mdtol=-1)


def _test_mask_from_init(scheme, mask_keyword):
    """
    Test initialisation of scheme.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    # Create a scheme with and without masking behaviour
    kwargs = {mask_keyword: True}
    default_scheme = scheme()
    masked_scheme = scheme(**kwargs)
    assert getattr(default_scheme, mask_keyword) is False
    assert getattr(masked_scheme, mask_keyword) is True

    n_lons_src = 6
    n_lats_src = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    cube = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    data = np.zeros([n_lats_src, n_lons_src])
    mask = np.zeros([n_lats_src, n_lons_src])
    mask[0, 0] = 1
    data = ma.array(data, mask=mask)
    cube.data = data

    # Create a regridder from the scheme.
    default_rg = default_scheme.regridder(cube, cube)
    masked_rg = masked_scheme.regridder(cube, cube)

    # Remove "use_" from the keyword to get the equivalent attr used on the regridder.
    regridder_attr = mask_keyword[4:]

    # Check that the mask stored on the regridder is correct.
    assert getattr(default_rg, regridder_attr) is None
    np.testing.assert_allclose(getattr(masked_rg, regridder_attr), mask)


def _test_mask_from_regridder(scheme, mask_keyword):
    """
    Test regridder method of the scheme.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    n_lons_src = 6
    n_lats_src = 4
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    cube = _grid_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True)
    data = np.zeros([n_lats_src, n_lons_src])
    mask = np.zeros([n_lats_src, n_lons_src])
    mask[0, 0] = 1
    data = ma.array(data, mask=mask)
    cube.data = data

    mask_different = np.zeros([n_lats_src, n_lons_src])
    mask_different[1, 2] = 1

    # Create a scheme without default masking behaviour.
    default_scheme = scheme()

    # Create a regridder from the mask on the cube.
    kwargs = {mask_keyword: True}
    rg_from_cube = default_scheme.regridder(cube, cube, **kwargs)

    # Create a regridder from a user supplied mask.
    kwargs_different = {mask_keyword: mask_different}
    rg_from_different = default_scheme.regridder(cube, cube, **kwargs_different)

    # Remove "use_" from the keyword to get the equivalent attr used on the regridder.
    regridder_attr = mask_keyword[4:]

    # Check that the mask stored on the regridder is correct.
    np.testing.assert_allclose(getattr(rg_from_cube, regridder_attr), mask)
    np.testing.assert_allclose(
        getattr(rg_from_different, regridder_attr), mask_different
    )


def _test_non_degree_crs(scheme):
    """Test regridding scheme is compatible with coordinates with non-degree units."""
    coord_system = OSGB()

    # This definition comes from a small section of real user data.
    n_lons_src = 2
    n_lats_src = 3
    lon_bounds = (-197500, -192500)
    lat_bounds = (1247500, 1237500)
    tm_cube = _grid_cube(
        n_lons_src,
        n_lats_src,
        lon_bounds,
        lat_bounds,
        circular=False,
        coord_system=coord_system,
        standard_names=["projection_x_coordinate", "projection_y_coordinate"],
        units="m",
    )
    data = np.arange(n_lats_src * n_lons_src).reshape([n_lats_src, n_lons_src])
    tm_cube.data = data

    n_lons_tgt = 12
    n_lats_tgt = 14
    lon_bounds_tgt = (-13, -12.8)
    lat_bounds_tgt = (60.5, 60.7)
    cube_tgt = _grid_cube(
        n_lons_tgt, n_lats_tgt, lon_bounds_tgt, lat_bounds_tgt, circular=True
    )

    result = tm_cube.regrid(cube_tgt, scheme())

    # Set expected results, this varies depending on the scheme.
    if scheme is ESMFAreaWeighted:
        expected_sum, expected_unmasked = 50.86147272655136, 21
    elif scheme is ESMFBilinear:
        expected_sum, expected_unmasked = 35.90837983047451, 13
    elif scheme is ESMFNearest:
        expected_sum, expected_unmasked = 490, 168

    # Check that the data is as expected.
    assert np.isclose(result.data.sum(), expected_sum)

    # Check that the number of masked points is as expected.
    assert (1 - result.data.mask).sum() == expected_unmasked
