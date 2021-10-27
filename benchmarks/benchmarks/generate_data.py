"""
Scripts for generating supporting data for benchmarking.

Data generated using iris-esmf-regrid should use
:func:`run_function_elsewhere`, which means that data is generated using a
fixed version of iris-esmf-regrid and a fixed environment, rather than those
that get changed when the benchmarking run checks out a new commit. The passed
python executable path in such a case should be within an environment that
supports iris-esmf-regrid and has iris-esmf-regrid installed via
``pip install -e``. See also :const:`DATA_GEN_PYTHON`.

Downstream use of data generated 'elsewhere' requires saving; usually in a
NetCDF file. Could also use pickling but there is a potential risk if the
benchmark sequence runs over two different Python versions.

"""
from inspect import getsource
from subprocess import CalledProcessError, check_output, run
from os import environ
from pathlib import Path
from textwrap import dedent

from iris import load_cube

# Allows the command line to pass in a python executable for use within
#  run_function_elsewhere()·
DATA_GEN_PYTHON = environ.get("DATA_GEN_PYTHON", "")
try:
    _ = check_output([DATA_GEN_PYTHON, "-c", "a = True"])
except (CalledProcessError, FileNotFoundError, PermissionError):
    error = (
        f"Expected valid python executable path from env variable "
        f"DATA_GEN_PYTHON. Got: {DATA_GEN_PYTHON}"
    )
    raise ValueError(error)


def run_function_elsewhere(python_exe, func_to_run, func_call_string):
    """
    Run a given function using an alternative python executable.

    This structure allows the function to be written natively, with only the
    function call needing to be written as a string.

    Parameters
    ----------
    python_exe : pathlib.Path or str
        Path to the alternative python executable.
    func_to_run : FunctionType
        The function object to be run using the alternative python executable.
    func_call_string: str
        A string that, when executed, will call the function with the desired arguments.

    Returns
    -------
    str
        The ``stdout`` from the run.

    """
    func_string = dedent(getsource(func_to_run))
    python_string = "\n".join([func_string, func_call_string])
    result = run([python_exe, "-c", python_string], capture_output=True, check=True)
    return result.stdout


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
        from iris import save
        from iris.coord_systems import RotatedGeogCS

        from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import _grid_cube

        save_path = kwargs.pop("save_path")

        if kwargs.pop("alt_coord_system"):
            kwargs["coord_system"] = RotatedGeogCS(0, 90, 90)

        cube = _grid_cube(*args, **kwargs)
        save(cube, save_path)

    save_dir = (Path(__file__).parent.parent / ".data").resolve()
    save_dir.mkdir(exist_ok=True)
    # TODO: caching? Currently written assuming overwrite every time.
    save_path = save_dir / "_grid_cube.nc"

    call_string = (
        "func("
        f"{n_lons}, {n_lats}, {lon_outer_bounds}, {lat_outer_bounds}, "
        f"{circular}, alt_coord_system={alt_coord_system}, "
        f"save_path='{save_path}'"
        ")"
    )
    _ = run_function_elsewhere(DATA_GEN_PYTHON, func, call_string)
    return_cube = load_cube(str(save_path))
    return return_cube
