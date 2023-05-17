Examples
========

To regrid a single Iris_ cube using an area-weighted conservative method::

    import iris
    from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
    from esmf_regrid.schemes import ESMFAreaWeighted

    # An example such a file can be found at:
    # https://github.com/SciTools/iris-test-data/blob/master/test_data/NetCDF/unstructured_grid/data_C4.nc
    with PARSE_UGRID_ON_LOAD.context():
        source_mesh_cube = iris.load_cube("mesh_cube.nc")

    # An example of such a file can be found at:
    # https://github.com/SciTools/iris-test-data/blob/master/test_data/NetCDF/global/xyt/SMALL_hires_wind_u_for_ipcc4.nc
    target_grid_cube = iris.load_cube("grid_cube.nc")

    result = source_mesh_cube.regrid(target_grid_cube, ESMFAreaWeighted())

Note that this scheme is flexible and it is also possible to regrid from
an unstructured cube to a structured cube as follows::

    result = target_grid_cube.regrid(source_mesh_cube, ESMFAreaWeighted())


.. _Iris: https://github.com/SciTools/iris