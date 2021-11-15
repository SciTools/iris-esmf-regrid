"""Unit tests for :mod:`esmf_regrid.experimental.io.save_regridder`."""

import pytest

from esmf_regrid.experimental.io import save_regridder


def test_invalid_type(tmp_path):
    """Test that `save_regridder` raises a TypeError where appropriate."""
    invalid_obj = None
    filename = tmp_path / "regridder.nc"
    with pytest.raises(TypeError):
        save_regridder(invalid_obj, filename)
