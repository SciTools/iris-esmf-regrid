"""Common conveniences for testing regridding schemes."""

from abc import ABC

import numpy as np
from numpy import ma
import pytest

from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)


class _CommonScheme(ABC):
    """A class containing tests for a scheme, configurable via subclassing."""

    # Set SCHEME to the desired scheme class.
    SCHEME = NotImplemented

    @pytest.mark.parametrize(
        "src_type,tgt_type", [("grid", "grid"), ("grid", "mesh"), ("mesh", "grid")]
    )
    @pytest.mark.parametrize(
        "full_mdtol", [False, True], ids=["no_mdtol", "full_mdtol"]
    )
    def test_cube_regrid(self, src_type, tgt_type, full_mdtol):
        """
        Test that the scheme class can be passed to a cubes regrid method.

        Checks that regridding occurs and that mdtol is used correctly.
        """
        kwargs = dict()
        if full_mdtol:
            kwargs["mdtol"] = 1
        scheme = self.SCHEME(**kwargs)

        n_lons_src = 6
        n_lons_tgt = 3
        n_lats_src = 4
        n_lats_tgt = 2
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        if src_type == "grid":
            src = _grid_cube(
                n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True
            )
            src_data = np.zeros([n_lats_src, n_lons_src])
            src_mask = np.zeros([n_lats_src, n_lons_src])
            src_mask[0, 0] = 1
        else:
            src = _gridlike_mesh_cube(n_lons_src, n_lats_src)
            src_data = np.zeros([n_lats_src * n_lons_src])
            src_mask = np.zeros([n_lats_src * n_lons_src])
            src_mask[0] = 1
        if tgt_type == "grid":
            tgt = _grid_cube(
                n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True
            )
            expected_data = np.zeros([n_lats_tgt, n_lons_tgt])
            expected_mask = np.zeros([n_lats_tgt, n_lons_tgt])
            expected_mask[0, 0] = 1
        else:
            tgt = _gridlike_mesh_cube(n_lons_tgt, n_lats_tgt)
            expected_data = np.zeros([n_lats_tgt * n_lons_tgt])
            expected_mask = np.zeros([n_lats_tgt * n_lons_tgt])
            expected_mask[0] = 1
        src_data = ma.array(src_data, mask=src_mask)
        src.data = src_data

        result = src.regrid(tgt, scheme)

        if full_mdtol:
            expected_data = ma.array(expected_data, mask=expected_mask)
        expected_cube = tgt.copy()
        expected_cube.data = expected_data
        assert expected_cube == result

    def test_invalid_mdtol(self):
        """Test erroring when mdtol is out of range."""
        match = "Value for mdtol must be in range 0 - 1, got "
        with pytest.raises(ValueError, match=match):
            _ = self.SCHEME(mdtol=2)
        with pytest.raises(ValueError, match=match):
            _ = self.SCHEME(mdtol=-1)

    @pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
    def test_mask_from_init(self, mask_keyword):
        """Test correct passing of mask arguments in init."""
        # Create a scheme with and without masking behaviour
        kwargs = {mask_keyword: True}
        default_scheme = self.SCHEME()
        masked_scheme = self.SCHEME(**kwargs)
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

    @pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
    def test_mask_from_regridder(self, mask_keyword):
        """Test correct passing of mask arguments in regridder."""
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
        default_scheme = self.SCHEME()

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
