"""Unit tests for :class:`esmf_regrid.schemes.ESMFBilinear`."""

import pytest

from esmf_regrid.schemes import ESMFBilinear
from esmf_regrid.tests.unit.schemes.__init__ import (
    _test_cube_regrid,
    _test_invalid_mdtol,
    _test_mask_from_init,
    _test_mask_from_regridder,
    _test_non_degree_crs,
)


@pytest.mark.parametrize(
    "src_type,tgt_type",
    [
        ("grid", "grid"),
        ("grid", "mesh"),
        ("grid", "just_mesh"),
        ("mesh", "grid"),
        ("mesh", "mesh"),
    ],
)
def test_cube_regrid(src_type, tgt_type):
    """
    Test that ESMFBilinear can be passed to a cubes regrid method.

    Checks that regridding occurs and that mdtol is used correctly.
    """
    _test_cube_regrid(ESMFBilinear, src_type, tgt_type)


def test_invalid_mdtol():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinear`.

    Checks that an error is raised when mdtol is out of range.
    """
    _test_invalid_mdtol(ESMFBilinear)


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_init(mask_keyword):
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFBilinear`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_init(ESMFBilinear, mask_keyword)


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_regridder(mask_keyword):
    """
    Test regridder method of :class:`esmf_regrid.schemes.ESMFBilinear`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_regridder(ESMFBilinear, mask_keyword)


def test_non_degree_crs():
    """Test for coordinates with non-degree units."""
    _test_non_degree_crs(ESMFBilinear)
