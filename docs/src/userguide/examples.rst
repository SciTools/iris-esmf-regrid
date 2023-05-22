Examples
========

Simple Regridding
-----------------

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
a structured cube to an unstructured cube as follows::

    result = target_grid_cube.regrid(source_mesh_cube, ESMFAreaWeighted())

Saving and Loading a Regridder
------------------------------
A regridder can be set up for reuse, this saves time performing the
computationally expensive initialisation process::

    from esmf_regrid.experimental.unstructured_scheme import MeshToGridESMFRegridder

    # Initialise the regridder with a source mesh and target grid.
    regridder = MeshToGridESMFRegridder(source_mesh_cube, target_grid_cube)

    # use the initialised regridder to regrid the data from the source cube
    # onto a cube with the same grid as `target_grid_cube`.
    result = regridder(source_mesh_cube)

To make use of this efficiency across sessions, we support the saving of
certain regridders. We can do this as follows::

    from esmf_regrid.experimental.io import load_regridder, save_regridder

    # Save the regridder.
    save_regridder(regridder, "saved_regridder.nc")

    # Load saved regridder.
    loaded_regridder = load_regridder("saved_regridder.nc")

    # Use loaded regridder.
    result = loaded_regridder(source_mesh_cube)

.. todo:
    Add more examples.

.. _Iris: https://github.com/SciTools/iris