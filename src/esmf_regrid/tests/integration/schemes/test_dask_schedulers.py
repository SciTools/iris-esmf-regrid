import contextlib
import dask
import distributed

from esmf_regrid.schemes import ESMFAreaWeighted
from esmf_regrid.tests.unit.schemes import _test_cube_regrid


@contextlib.contextmanager
def distributed_context():
    _distributed_client = distributed.Client()
    yield
    _distributed_client.close()


def test_distributed_scheduler():
    with distributed_context():
        _test_cube_regrid(ESMFAreaWeighted, "grid", "grid")


def test_processes_scheduler():
    with dask.config.set(scheduler="processes"):
        _test_cube_regrid(ESMFAreaWeighted, "grid", "grid")


def test_threads_scheduler():
    with dask.config.set(scheduler="threads"):
        _test_cube_regrid(ESMFAreaWeighted, "grid", "grid")
