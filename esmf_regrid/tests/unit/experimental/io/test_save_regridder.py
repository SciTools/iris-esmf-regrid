"""Unit tests for :mod:`esmf_regrid.experimental.io.save_regridder`."""

import pytest

from esmf_regrid.experimental.io import save_regridder
from esmf_regrid.tests import temp_filename


def test_invalid_type():
    """Test that `save_regridder` raises a TypeError where appropriate."""
    invalid_obj = None
    with temp_filename(suffix=".nc") as filename:
        with pytest.raises(TypeError):
            save_regridder(invalid_obj, filename)
