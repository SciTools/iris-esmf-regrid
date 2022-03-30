"""Unit tests for :class:`esmf_regrid._esmf_sdo.RefinedGridInfo`."""

import numpy as np
from numpy import ma

from esmf_regrid.esmf_regridder import RefinedGridInfo, Regridder
from esmf_regrid.experimental.unstructured_regrid import MeshInfo
from esmf_regrid.tests import make_grid_args
from esmf_regrid.tests.unit.experimental.unstructured_regrid.test_MeshInfo import (
    _make_small_mesh_args,
)


def test_expanded_lons_with_mesh():
    """
    Basic test for regridding with :meth:`~esmf_regrid.esmf_regridder.RefinedGridInfo.make_esmf_field`.

    Mirrors the tests in :func:`~esmf_regrid.tests.unit.experimental.unstructured_regrid.test_MeshInfo.test_regrid_with_mesh`
    but with slightly different expected values due to increased accuracy.
    """
    mesh_args = _make_small_mesh_args()
    mesh = MeshInfo(*mesh_args)

    grid_args = make_grid_args(2, 3)
    grid = RefinedGridInfo(*grid_args[2:4], resolution=4)

    mesh_to_grid_regridder = Regridder(mesh, grid)
    mesh_input = np.array([3, 2])
    grid_output = mesh_to_grid_regridder.regrid(mesh_input)
    expected_grid_output = np.array(
        [
            [2.671534474734418, 3.0],
            [2.088765949748455, 2.922517356506756],
            [2.0, 2.340882413622917],
        ]
    )
    assert ma.allclose(expected_grid_output, grid_output)

    grid_to_mesh_regridder = Regridder(grid, mesh)
    grid_input = np.array([[0, 0], [1, 0], [2, 1]])
    mesh_output = grid_to_mesh_regridder.regrid(grid_input)
    expected_mesh_output = np.array([0.14117205318254747, 1.1976140197893996])
    assert ma.allclose(expected_mesh_output, mesh_output)

    def _give_extra_dims(array):
        result = np.stack([array, array + 1])
        result = np.stack([result, result + 10, result + 100])
        return result

    extra_dim_mesh_input = _give_extra_dims(mesh_input)
    extra_dim_grid_output = mesh_to_grid_regridder.regrid(extra_dim_mesh_input)
    extra_dim_expected_grid_output = _give_extra_dims(expected_grid_output)
    assert ma.allclose(extra_dim_expected_grid_output, extra_dim_grid_output)

    extra_dim_grid_input = _give_extra_dims(grid_input)
    extra_dim_mesh_output = grid_to_mesh_regridder.regrid(extra_dim_grid_input)
    extra_dim_expected_mesh_output = _give_extra_dims(expected_mesh_output)
    assert ma.allclose(extra_dim_expected_mesh_output, extra_dim_mesh_output)


def test_expanded_lats_with_mesh():
    """Basic test for regridding with :meth:`~esmf_regrid.esmf_regridder.RefinedGridInfo.make_esmf_field`."""
    mesh_args = _make_small_mesh_args()
    mesh = MeshInfo(*mesh_args)

    grid = RefinedGridInfo(np.array([0, 5, 10]), np.array([-90, 90]), resolution=4)

    mesh_to_grid_regridder = Regridder(mesh, grid)
    mesh_input = np.array([3, 2])
    grid_output = mesh_to_grid_regridder.regrid(mesh_input)
    expected_grid_output = np.array(
        [
            [2.2024695514629724, 2.4336888097502642],
        ]
    )
    assert ma.allclose(expected_grid_output, grid_output)

    grid_to_mesh_regridder = Regridder(grid, mesh)
    grid_input = np.array([[1, 2]])
    mesh_output = grid_to_mesh_regridder.regrid(grid_input)
    expected_mesh_output = np.array([1.7480791292591336, 1.496070008348207])
    assert ma.allclose(expected_mesh_output, mesh_output)

    def _give_extra_dims(array):
        result = np.stack([array, array + 1])
        result = np.stack([result, result + 10, result + 100])
        return result

    extra_dim_mesh_input = _give_extra_dims(mesh_input)
    extra_dim_grid_output = mesh_to_grid_regridder.regrid(extra_dim_mesh_input)
    extra_dim_expected_grid_output = _give_extra_dims(expected_grid_output)
    assert ma.allclose(extra_dim_expected_grid_output, extra_dim_grid_output)

    extra_dim_grid_input = _give_extra_dims(grid_input)
    extra_dim_mesh_output = grid_to_mesh_regridder.regrid(extra_dim_grid_input)
    extra_dim_expected_mesh_output = _give_extra_dims(expected_mesh_output)
    assert ma.allclose(extra_dim_expected_mesh_output, extra_dim_mesh_output)
