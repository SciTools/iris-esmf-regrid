"""Unit tests for :mod:`esmf_regrid.experimental.partition`."""

import dask.array as da
import esmpy
import numpy as np
import pytest

from esmf_regrid import ESMFAreaWeighted, ESMFBilinear, ESMFNearest
from esmf_regrid.experimental.partition import Partition
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)
from esmf_regrid.tests.unit.schemes.test_regrid_rectilinear_to_rectilinear import (
    _make_full_cubes,
)


def test_Partition(tmp_path):
    """Test basic implementation of Partition class."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150 * 500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    blocks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    partition = Partition(src, tgt, scheme, files, explicit_src_blocks=blocks)

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected


def test_Partition_block_api(tmp_path):
    """Test API for controlling block shape for Partition class."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    num_src_chunks = (5, 1)
    partition = Partition(src, tgt, scheme, files, num_src_chunks=num_src_chunks)

    expected_chunks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    assert partition.src_blocks == expected_chunks

    src_chunks = (100, 150)
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_chunks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    assert partition.src_blocks == expected_chunks

    src_chunks = ((100, 100, 100, 100, 100), (150,))
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_chunks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    assert partition.src_blocks == expected_chunks

    src.data = da.from_array(src.data, chunks=src_chunks)
    partition = Partition(src, tgt, scheme, files, use_dask_src_chunks=True)

    expected_chunks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    assert partition.src_blocks == expected_chunks


def test_Partition_mesh_src(tmp_path):
    """Test Partition class when the source has a mesh."""
    src = _gridlike_mesh_cube(150, 500)
    src.data = np.arange(150 * 500)
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    src_chunks = (15000,)
    with pytest.raises(NotImplementedError):
        _ = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    # TODO: when mesh partitioning becomes possible, uncomment.
    # expected_src_chunks = [[[0, 15000]], [[15000, 30000]], [[30000, 45000]], [[45000, 60000]], [[60000, 75000]]]
    # assert partition.src_blocks == expected_src_chunks
    #
    # partition.generate_files()
    #
    # result = partition.apply_regridders(src)
    # expected = src.regrid(tgt, scheme)
    # assert np.allclose(result.data, expected.data)
    # assert result == expected


def test_Partition_curv_src(tmp_path):
    """Test Partition class when the source has a curvilinear grid."""
    src = _curvilinear_cube(150, 500, (-180, 180), (-90, 90))
    src.data = np.arange(150 * 500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    src_chunks = (100, 150)
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    expected_src_chunks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]
    assert partition.src_blocks == expected_src_chunks

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected


def test_Partition_bilinear(tmp_path):
    """Test Partition class for bilinear regridding."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150 * 500).reshape([500, 150])
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    src_chunks = (100, 150)

    bad_scheme = ESMFBilinear()
    with pytest.raises(ValueError):
        _ = Partition(src, tgt, bad_scheme, files, src_chunks=src_chunks)

    # The pole_method must be NONE for bilinear regridding partitions to work.
    scheme = ESMFBilinear(esmf_args={"pole_method": esmpy.PoleMethod.NONE})

    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected


def test_Partition_mesh_tgt(tmp_path):
    """Test Partition class when the target has a mesh."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    src.data = np.arange(150 * 500).reshape([500, 150])
    tgt = _gridlike_mesh_cube(16, 36)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)

    src_chunks = (100, 150)
    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)

    partition.generate_files()

    result = partition.apply_regridders(src)
    expected = src.regrid(tgt, scheme)
    assert np.allclose(result.data, expected.data)
    assert result == expected


def test_conflicting_chunks(tmp_path):
    """Test error handling of Partition class."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    scheme = ESMFAreaWeighted(mdtol=1)
    num_src_chunks = (5, 1)
    src_chunks = (100, 150)
    blocks = [
        [[0, 100], [0, 150]],
        [[100, 200], [0, 150]],
        [[200, 300], [0, 150]],
        [[300, 400], [0, 150]],
        [[400, 500], [0, 150]],
    ]

    with pytest.raises(ValueError):
        _ = Partition(
            src,
            tgt,
            scheme,
            files,
            src_chunks=src_chunks,
            num_src_chunks=num_src_chunks,
        )
    with pytest.raises(ValueError):
        _ = Partition(
            src, tgt, scheme, files, src_chunks=src_chunks, explicit_src_blocks=blocks
        )
    with pytest.raises(ValueError):
        _ = Partition(src, tgt, scheme, files)
    with pytest.raises(TypeError):
        _ = Partition(src, tgt, scheme, files, use_dask_src_chunks=True)
    with pytest.raises(ValueError):
        _ = Partition(src, tgt, scheme, files[:-1], src_chunks=src_chunks)


def test_multidimensional_cube(tmp_path):
    """Test Partition class when the source has a multidimensional cube."""
    src_cube, tgt_grid, expected_cube = _make_full_cubes()
    files = [tmp_path / f"partial_{x}.nc" for x in range(4)]
    scheme = ESMFAreaWeighted(mdtol=1)
    chunks = (2, 3)

    partition = Partition(src_cube, tgt_grid, scheme, files, src_chunks=chunks)

    partition.generate_files()

    result = partition.apply_regridders(src_cube)

    # Lenient check for data.
    assert np.allclose(result.data, expected_cube.data)

    # Check metadata and coords.
    result.data = expected_cube.data
    assert result == expected_cube


def test_save_incomplete(tmp_path):
    """Test Partition class when a limited number of files are saved."""
    src = _grid_cube(150, 500, (-180, 180), (-90, 90), circular=True)
    tgt = _grid_cube(16, 36, (-180, 180), (-90, 90), circular=True)

    files = [tmp_path / f"partial_{x}.nc" for x in range(5)]
    src_chunks = (100, 150)
    scheme = ESMFAreaWeighted(mdtol=1)
    num_initial_chunks = 3
    expected_files = files[:num_initial_chunks]

    partition = Partition(src, tgt, scheme, files, src_chunks=src_chunks)
    with pytest.raises(OSError):
        _ = partition.apply_regridders(src, allow_incomplete=True)

    partition.generate_files(files_to_generate=num_initial_chunks)
    assert partition.saved_files == expected_files

    expected_array_partial = np.ma.zeros([36, 16])
    expected_array_partial[22:] = np.ma.masked

    with pytest.raises(OSError):
        _ = partition.apply_regridders(src)
    partial_result = partition.apply_regridders(src, allow_incomplete=True)
    assert np.ma.allclose(partial_result.data, expected_array_partial)

    loaded_partition = Partition(
        src, tgt, scheme, files, src_chunks=src_chunks, saved_files=expected_files
    )

    with pytest.raises(OSError):
        _ = loaded_partition.apply_regridders(src)
    partial_result_2 = partition.apply_regridders(src, allow_incomplete=True)
    assert np.ma.allclose(partial_result_2.data, expected_array_partial)

    loaded_partition.generate_files()

    result = loaded_partition.apply_regridders(src)
    expected_array = np.ma.zeros([36, 16])
    assert np.ma.allclose(result.data, expected_array)


def test_nearest_invalid(tmp_path):
    """Test Partition class when initialised with an invalid scheme."""
    src_cube, tgt_grid, _ = _make_full_cubes()
    files = [tmp_path / f"partial_{x}.nc" for x in range(4)]
    scheme = ESMFNearest()
    chunks = (2, 3)

    with pytest.raises(NotImplementedError):
        _ = Partition(src_cube, tgt_grid, scheme, files, src_chunks=chunks)


def test_Partition_repr(tmp_path):
    """Test repr of Partition instance."""
    src_cube, tgt_grid, _ = _make_full_cubes()
    files = [tmp_path / f"partial_{x}.nc" for x in range(4)]
    scheme = ESMFAreaWeighted()
    chunks = (2, 3)

    partition = Partition(src_cube, tgt_grid, scheme, files, src_chunks=chunks)

    expected_repr = (
        "Partition(src=<iris 'Cube' of air_temperature / (K) "
        "(height: 2; latitude: 3; time: 4; longitude: 5; -- : 6)>, "
        "tgt=<iris 'Cube' of unknown / (unknown) (latitude: 5; longitude: 3)>, "
        "scheme=ESMFAreaWeighted(mdtol=0, use_src_mask=False, use_tgt_mask=False, esmf_args={}), "
        "num file_names=4,num saved_files=0)"
    )
    assert repr(partition) == expected_repr
