"""
Scripts for generating supporting data for benchmarking.

Data generated using iris-esmf-regrid should use
:func:`run_function_elsewhere`, which means that data is generated using a
fixed version of iris-esmf-regrid and a fixed environment, rather than those
that get changed when the benchmarking run checks out a new commit.

The python executable for :func:`run_function_elsewhere` is provided via the
``DATA_GEN_PYTHON`` environment variable, and should be the path of an
executable within an environment that supports iris-esmf-regrid and has
iris-esmf-regrid installed via ``pip install -e``.

Downstream use of data generated 'elsewhere' requires saving; usually in a
NetCDF file. Could also use pickling but there is a potential risk if the
benchmark sequence runs over two different Python versions.

"""
from inspect import getsource
from subprocess import CalledProcessError, check_output, run
from os import environ
from textwrap import dedent

from iris import load_cube

PYTHON_EXE = environ.get("DATA_GEN_PYTHON", "")
try:
    _ = check_output([PYTHON_EXE, "-c", "a = True"])
except (CalledProcessError, FileNotFoundError, PermissionError):
    error = (
        f"Expected valid python executable path from env variable "
        f"DATA_GEN_PYTHON. Got: {PYTHON_EXE}"
    )
    raise ValueError(error)


def run_function_elsewhere(python_exe, func_to_run, func_call_string):
    """
    Run a given function using an alternative python executable.

    This structure allows the function to be written natively, with only the
    function call needing to be written as a string.

    Uses print (stdout) as the only available 'return'.

    Parameters
    ----------
    python_exe : pathlib.Path or str
        Path to the alternative python executable.
    func_to_run : FunctionType
        The function object to be run using the alternative python executable.
    func_call_string: str
        A string that, when executed, will call the function with the desired arguments.

    """
    func_string = dedent(getsource(func_to_run))
    python_string = "\n".join([func_string, func_call_string])
    result = run([python_exe, "-c", python_string], capture_output=True, check=True)
    return result.stdout.decode().strip()


def _grid_cube(
    n_lons,
    n_lats,
    lon_outer_bounds,
    lat_outer_bounds,
    circular=False,
    alt_coord_system=False,
):
    """Wrapper for calling _grid_cube using an alternative python executable."""

    def func(*args, **kwargs):
        """
        Prep and run _grid_cube, saving to a NetCDF file.

        Saving to a file allows the original python executable to pick back up.

        Remember that all arguments must work as strings, hence the fresh
        construction of a ``coord_system`` within the function.

        """
        from pathlib import Path

        from iris import save
        from iris.coord_systems import RotatedGeogCS

        from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube

        save_dir = (Path().parent.parent / ".data").resolve()
        save_dir.mkdir(exist_ok=True)
        # TODO: caching? Currently written assuming overwrite every time.
        save_path = save_dir / "_grid_cube.nc"

        if kwargs.pop("alt_coord_system"):
            kwargs["coord_system"] = RotatedGeogCS(0, 90, 90)

        cube = _grid_cube(*args, **kwargs)
        save(cube, str(save_path))
        # Print the path for downstream use - is returned by run_function_elsewhere().
        print(save_path)

    call_string = (
        "func("
        f"{n_lons}, {n_lats}, {lon_outer_bounds}, {lat_outer_bounds}, "
        f"{circular}, alt_coord_system={alt_coord_system}"
        ")"
    )
    cube_file = run_function_elsewhere(PYTHON_EXE, func, call_string)
    return_cube = load_cube(cube_file)
    return return_cube
