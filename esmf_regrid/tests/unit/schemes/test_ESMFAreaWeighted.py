"""Unit tests for :class:`esmf_regrid.schemes.ESMFAreaWeighted`."""

import pytest

from esmf_regrid.schemes import ESMFAreaWeighted
from esmf_regrid.tests.unit.schemes.__init__ import (
    _test_cube_regrid,
    _test_invalid_mdtol,
    _test_mask_from_init,
    _test_mask_from_regridder,
)


@pytest.mark.parametrize(
    "src_type,tgt_type", [("grid", "grid"), ("grid", "mesh"), ("mesh", "grid")]
)
def test_cube_regrid(src_type, tgt_type):
    """
    Test that ESMFAreaWeighted can be passed to a cubes regrid method.

    Checks that regridding occurs and that mdtol is used correctly.
    """
    _test_cube_regrid(ESMFAreaWeighted, src_type, tgt_type)


def test_invalid_mdtol():
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFAreaWeighted`.

    Checks that an error is raised when mdtol is out of range.
    """
    _test_invalid_mdtol(ESMFAreaWeighted)


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_init(mask_keyword):
    """
    Test initialisation of :class:`esmf_regrid.schemes.ESMFAreaWeighted`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_init(ESMFAreaWeighted, mask_keyword)


@pytest.mark.parametrize("mask_keyword", ["use_src_mask", "use_tgt_mask"])
def test_mask_from_regridder(mask_keyword):
    """
    Test regridder method of :class:`esmf_regrid.schemes.ESMFAreaWeighted`.

    Checks that use_src_mask and use_tgt_mask are passed down correctly.
    """
    _test_mask_from_regridder(ESMFAreaWeighted, mask_keyword)
