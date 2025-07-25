"""Unit tests for :mod:`esmf_regrid.experimental.partition`."""
import numpy as np
from esmf_regrid import ESMFAreaWeighted
from esmf_regrid.experimental.partition import Partition, Partition2

from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)

def test_Partition(tmp_path):
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150*500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)
    chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    # TODO: consider different ways we could specify this in the API:
    # chunks = 5
    # chunks = (5, None)
    # chunks = None  # (use dask chunks)

    partition = Partition(src, tgt, scheme, files, chunks)

    partition.generate_files()

    result = partition.apply_regridders(src)

    expected = src.regrid(tgt, scheme)

    assert result == expected
    assert np.array_equal(result.data, expected.data)

def test_Partition2(tmp_path):
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150*500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)
    chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    # TODO: consider different ways we could specify this in the API:
    # chunks = 5
    # chunks = (5, None)
    # chunks = None  # (use dask chunks)

    partition = Partition2(src, tgt, scheme, files, chunks)

    partition.generate_files()

    result = partition.apply_regridders(src)


    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected
