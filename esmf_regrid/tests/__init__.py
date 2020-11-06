"""Common testing infrastructure."""

import pathlib


# Base directory of the test results.
_RESULT_PATH = pathlib.Path(__file__).parent.resolve() / "results"


def get_result_path(relative_path, unit=True):
    """
    Form the absolute path to the results test file.

    Parameters
    ----------
    relative_path : str, path, iterable str/path
        The relative path to the target test file.
    unit : bool, optional
        Specify whether the `relative_path` is for a unit test.
        Default is True.

    Returns
    -------
    Path
        The absolute result path.

    """
    if isinstance(relative_path, str):
        relative_path = pathlib.Path(relative_path)

    if not isinstance(relative_path, pathlib.PurePath):
        relative_path = pathlib.Path(*relative_path)

    if unit:
        relative_path = pathlib.Path("unit") / relative_path

    result = _RESULT_PATH / relative_path

    return result.resolve(strict=True)
