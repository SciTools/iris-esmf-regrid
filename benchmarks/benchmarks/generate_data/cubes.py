"""Scripts for generating supporting Cubes for benchmarking."""

import re

from iris import load_cube

from esmf_regrid import _load_context

from . import BENCHMARK_DATA, REUSE_DATA, run_function_elsewhere


def _grid_cube(
    n_lons,
    n_lats,
    lon_outer_bounds,
    lat_outer_bounds,
    circular=False,
    alt_coord_system=False,
):
    """Call _grid_cube via :func:`run_function_elsewhere`."""

    def external(*args, **kwargs):
        """Prep and call _grid_cube, saving to a NetCDF file.

        Saving to a file allows the original python executable to pick back up.

        Remember that all arguments must work as strings, hence the fresh
        construction of a ``coord_system`` within the function.

        """
        from iris import save
        from iris.coord_systems import RotatedGeogCS

        from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
            _grid_cube as original,
        )

        save_path = kwargs.pop("save_path")

        if kwargs.pop("alt_coord_system"):
            kwargs["coord_system"] = RotatedGeogCS(0, 90, 90)

        cube = original(*args, **kwargs)
        save(cube, save_path)

    file_name_sections = [
        "_grid_cube",
        n_lons,
        n_lats,
        lon_outer_bounds,
        lat_outer_bounds,
        circular,
        alt_coord_system,
    ]
    file_name = "_".join(str(section) for section in file_name_sections)
    # Remove 'unsafe' characters.
    file_name = re.sub(r"\W+", "", file_name)
    save_path = (BENCHMARK_DATA / file_name).with_suffix(".nc")

    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            external,
            n_lons,
            n_lats,
            lon_outer_bounds,
            lat_outer_bounds,
            circular,
            alt_coord_system=alt_coord_system,
            save_path=str(save_path),
        )

    return_cube = load_cube(str(save_path))
    return return_cube


def _curvilinear_cube(
    n_lons,
    n_lats,
    lon_outer_bounds,
    lat_outer_bounds,
):
    """Call _curvilinear_cube via :func:`run_function_elsewhere`."""

    def external(*args, **kwargs):
        """Prep and call _curvilinear_cube, saving to a NetCDF file.

        Saving to a file allows the original python executable to pick back up.

        Remember that all arguments must work as strings.

        """
        from iris import save

        from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
            _curvilinear_cube as original,
        )

        save_path = kwargs.pop("save_path")

        cube = original(*args, **kwargs)
        save(cube, save_path)

    file_name_sections = [
        "_curvilinear_cube",
        n_lons,
        n_lats,
        lon_outer_bounds,
        lat_outer_bounds,
    ]
    file_name = "_".join(str(section) for section in file_name_sections)
    # Remove 'unsafe' characters.
    file_name = re.sub(r"\W+", "", file_name)
    save_path = (BENCHMARK_DATA / file_name).with_suffix(".nc")

    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            external,
            n_lons,
            n_lats,
            lon_outer_bounds,
            lat_outer_bounds,
            save_path=str(save_path),
        )

    return_cube = load_cube(str(save_path))
    return return_cube


def _gridlike_mesh_cube(n_lons, n_lats):
    """Call _gridlike_mesh via :func:`run_function_elsewhere`."""

    def external(*args, **kwargs):
        """Prep and call _gridlike_mesh, saving to a NetCDF file.

        Saving to a file allows the original python executable to pick back up.

        """
        from iris import save

        from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
            _gridlike_mesh_cube as original,
        )

        save_path = kwargs.pop("save_path")

        cube = original(*args, **kwargs)
        save(cube, save_path)

    save_path = BENCHMARK_DATA / f"_mesh_cube_{n_lons}_{n_lats}.nc"

    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            external,
            n_lons,
            n_lats,
            save_path=str(save_path),
        )

    with _load_context():
        return_cube = load_cube(str(save_path))
    return return_cube
