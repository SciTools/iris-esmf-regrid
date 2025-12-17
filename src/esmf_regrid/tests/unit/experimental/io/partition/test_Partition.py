"""Unit tests for :mod:`esmf_regrid.experimental.partition`."""

import dask.array as da
import numpy as np
import pytest

import esmf_regrid
print(esmf_regrid.__file__)
from esmf_regrid import ESMFAreaWeighted
from esmf_regrid.experimental.partition import Partition

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

    partition = Partition(src, tgt, scheme, files, explicit_src_chunks=chunks)

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected

def test_Partition_block_api(tmp_path):
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150*500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)
    num_src_chunks = (5, 1)
    partition = Partition(src, tgt, scheme, files, num_src_chunks=num_src_chunks)

    expected_chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    assert partition.src_chunks == expected_chunks

    src_chunks = (100, 150)
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    assert partition.src_chunks == expected_chunks


    src_chunks = ((100, 100, 100, 100, 100), (150,))
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    assert partition.src_chunks == expected_chunks

    src.data = da.from_array(src.data, chunks=src_chunks)
    partition = Partition(src, tgt, scheme, files, use_dask_src_chunks=True)

    expected_chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    assert partition.src_chunks == expected_chunks

def test_Partition_mesh_src(tmp_path):
    src = _gridlike_mesh_cube(150, 500)
    src.data = np.arange(150*500)
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    src_chunks = (15000,)
    with pytest.raises(NotImplementedError):
        partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    # TODO when mesh partitioning becomes possible, uncomment.
    # expected_src_chunks = [[[0, 15000]], [[15000, 30000]], [[30000, 45000]], [[45000, 60000]], [[60000, 75000]]]
    # assert partition.src_chunks == expected_src_chunks
    #
    # partition.generate_files()
    #
    # result = partition.apply_regridders(src)
    # expected = src.regrid(tgt, scheme)
    # assert np.allclose(result.data, expected.data)
    # assert result == expected

def test_Partition_curv_src(tmp_path):
    src = _curvilinear_cube(150, 500, (-180, 180), (-90, 90))
    src.data = np.arange(150*500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    src_chunks = (100, 150)
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_src_chunks = [[[0, 100], [0, 150]], [[100, 200], [0, 150]], [[200, 300], [0, 150]], [[300, 400], [0, 150]], [[400, 500], [0, 150]]]
    assert partition.src_chunks == expected_src_chunks

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected
