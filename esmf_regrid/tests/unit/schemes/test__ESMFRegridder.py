"""Unit tests for :class:`esmf_regrid.schemes._ESMFRegridder`."""

import pytest

from esmf_regrid.schemes import _ESMFRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _grid_cube,
)


def test_invalid_method():
    """
    Test initialisation of :class:`esmf_regrid.schemes._ESMFRegridder`.

    Checks that an error is raised when the method is invalid.
    """
    n_lons = 6
    n_lats = 5
    lon_bounds = (-180, 180)
    lat_bounds = (-90, 90)
    src = tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

    with pytest.raises(NotImplementedError):
        _ = _ESMFRegridder(src, tgt, method="other")
