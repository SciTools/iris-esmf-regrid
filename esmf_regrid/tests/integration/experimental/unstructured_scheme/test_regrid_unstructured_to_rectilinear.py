"""Integration tests for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`."""


import os

import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy as np

from esmf_regrid.experimental.unstructured_scheme import (
    regrid_unstructured_to_rectilinear,
)


def test_real_data():
    """
    Test for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`.

    Tests with cubes derived from realistic data.
    """
    # Load source cube.
    test_data_dir = iris.config.TEST_DATA_DIR
    src_fn = os.path.join(
        test_data_dir, "NetCDF", "unstructured_grid", "lfric_surface_mean.nc"
    )
    with PARSE_UGRID_ON_LOAD.context():
        src = iris.load_cube(src_fn, "rainfall_flux")

    # Load target grid cube.
    tgt_fn = os.path.join(
        test_data_dir, "NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"
    )
    tgt = iris.load_cube(tgt_fn)

    # Perform regridding.
    result = regrid_unstructured_to_rectilinear(src, tgt)

    # Check data.
    assert result.shape == (1, 160, 320)
    assert np.isclose(result.data.mean(), 2.93844e-5)
    assert np.isclose(result.data.std(), 2.71724e-5)

    # Check metadata.
    assert result.metadata == src.metadata
    assert result.coord("time") == src.coord("time")
    assert result.coord("latitude") == tgt.coord("latitude")
    assert result.coord("longitude") == tgt.coord("longitude")
    assert result.coord_dims("time") == (0,)
    assert result.coord_dims("latitude") == (1,)
    assert result.coord_dims("longitude") == (2,)
