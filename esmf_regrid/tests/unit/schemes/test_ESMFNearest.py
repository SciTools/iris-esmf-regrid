"""Unit tests for :class:`esmf_regrid.schemes.ESMFNearest`."""

import pytest

from esmf_regrid.schemes import ESMFNearest
from ._common_scheme import _CommonScheme


class TestNearest(_CommonScheme):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFNearest`."""

    METHOD = ESMFNearest

    @pytest.mark.parametrize(
        "src_type,tgt_type", [("grid", "grid"), ("grid", "mesh"), ("mesh", "grid")]
    )
    def test_cube_regrid(self, src_type, tgt_type):
        """
        Test that :class:`esmf_regrid.schemes.ESMFNearest` can be passed to a cubes regrid method.

        Checks that regridding occurs.
        """
        # False for the full_mdtol parameter.
        super().test_cube_regrid(src_type, tgt_type, False)

    def test_invalid_mdtol(self):
        """Test initialisation of the method class - disabled for this subclass."""
        pytest.skip("mdtol inappropriate for Nearest scheme.")
