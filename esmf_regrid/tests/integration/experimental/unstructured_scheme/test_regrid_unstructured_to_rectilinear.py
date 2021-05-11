"""Integration tests for :func:`esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear`."""


import os
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD


def test_real_data():
    test_data_dir = iris.config.TEST_DATA_DIR
    fn = os.path.join(test_data_dir, "test_data", "NetCDF", "unstructured_grid", "lfric_ngvat_3D_snow_pseudo_levels_1t_face_half_levels_main_snow_layer_temp.nc")
    with PARSE_UGRID_ON_LOAD.context():
        cubes = iris.load(fn)
        cube = iris.load_cube(fn, "snow_layer_temperature")
