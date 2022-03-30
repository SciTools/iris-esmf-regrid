"""Unit tests for :class:`esmf_regrid.experimental.unstructured_regrid.MeshInfo`."""

import numpy as np
from numpy import ma

from esmf_regrid.esmf_regridder import GridInfo, Regridder
from esmf_regrid.experimental.unstructured_regrid import MeshInfo
from esmf_regrid.tests import get_result_path, make_grid_args


def _make_small_mesh_args():
    ugrid_node_coords = np.array(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [10.0, 20.0]]
    )
    ugrid_face_node_connectivity = ma.array(
        [[0, 2, 3, -1], [3, 0, 1, 4]],
        mask=np.array([[0, 0, 0, 1], [0, 0, 0, 0]]),
    )
    node_start_index = 0
    return ugrid_node_coords, ugrid_face_node_connectivity, node_start_index


def test_make_mesh():
    """Basic test for creating :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
    coords, nodes, _ = _make_small_mesh_args()
    mesh_0 = MeshInfo(coords, nodes, 0)
    esmf_mesh_0 = mesh_0.make_esmf_field()
    esmf_mesh_0.data[:] = 0

    relative_path = (
        "experimental",
        "unstructured_regrid",
        "test_MeshInfo",
        "small_mesh.txt",
    )
    fname = get_result_path(relative_path)
    with open(fname) as file:
        expected_repr = file.read()

    one_indexed_nodes = nodes + 1
    mesh_1 = MeshInfo(coords, one_indexed_nodes, 1)
    esmf_mesh_1 = mesh_1.make_esmf_field()
    esmf_mesh_1.data[:] = 0

    assert esmf_mesh_0.__repr__() == esmf_mesh_1.__repr__() == expected_repr


def test_regrid_with_mesh():
    """Basic test for regridding with :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
    mesh_args = _make_small_mesh_args()
    mesh = MeshInfo(*mesh_args)

    grid_args = make_grid_args(2, 3)
    grid = GridInfo(*grid_args)

    mesh_to_grid_regridder = Regridder(mesh, grid)
    mesh_input = np.array([3, 2])
    grid_output = mesh_to_grid_regridder.regrid(mesh_input)
    expected_grid_output = np.array(
        [
            [2.671294712940605, 3.0],
            [2.0885553467353097, 2.9222786250561574],
            [2.0, 2.3397940801753307],
        ]
    )
    assert ma.allclose(expected_grid_output, grid_output)

    grid_to_mesh_regridder = Regridder(grid, mesh)
    grid_input = np.array([[0, 0], [1, 0], [2, 1]])
    mesh_output = grid_to_mesh_regridder.regrid(grid_input)
    expected_mesh_output = np.array([0.1408245341331448, 1.19732762534643])
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


def test_regrid_bilinear_with_mesh():
    """Basic test for regridding with :meth:`~esmf_regrid.esmf_regridder.GridInfo.make_esmf_field`."""
    # We create a mesh with the following shape:
    # 20     ,+                  ,+ 4
    #       / |  with nodes:    / |
    # 10  +' ,+             1 +' ,+ 3
    #     | / |               | / |
    #  0  +---+             0 +---+ 2
    #     0  10
    mesh_args = _make_small_mesh_args()
    elem_coords = np.array([[5, 0], [5, 10]])
    node_mesh = MeshInfo(*mesh_args, location="node")
    face_mesh = MeshInfo(*mesh_args, elem_coords=elem_coords, location="face")

    # We create a grid with the following shape:
    # 20/3 +---+
    #      |   |
    # 10/3 +---+
    #      |   |
    #  0   +---+
    #      0  10
    grid_args = [ar * 2 for ar in make_grid_args(2, 3)]
    grid = GridInfo(*grid_args, center=True)

    mesh_to_grid_regridder = Regridder(node_mesh, grid, method="bilinear")
    mesh_input = np.arange(5)
    grid_output = mesh_to_grid_regridder.regrid(mesh_input)
    # For a flat surface, we would expect the fractional part of these values
    # to be either 1/3 or 2/3. Since the actual surface lies on a sphere, and
    # due to the way ESMF calculates these values, the expected output is
    # slightly different. It's worth noting that the finer the resolution, the
    # closer these numbers are. Since the grids/meshes lie on coarse steps of
    # about 10 degrees, we can expect most cases to behave more similarly to
    # bilinear regridders on flat surfaces.
    expected_grid_output = np.array(
        [
            [0.0, 2.0],
            [0.6662902773937054, 2.6662902773105808],
            [-1, 3.333709722689418],
        ]
    )
    expected_grid_mask = np.array([[0, 0], [0, 0], [1, 0]])
    expected_grid_output = ma.array(expected_grid_output, mask=expected_grid_mask)
    assert ma.allclose(expected_grid_output, grid_output)

    grid_to_mesh_regridder = Regridder(grid, node_mesh, method="bilinear")
    grid_input = np.array([[0, 0], [1, 0], [2, 1]])
    mesh_output = grid_to_mesh_regridder.regrid(grid_input)
    expected_mesh_output = ma.array([0.0, 1.5, 0.0, 0.5, -1], mask=[0, 0, 0, 0, 1])
    assert ma.allclose(expected_mesh_output, mesh_output)

    grid_to_face_mesh_regridder = Regridder(grid, face_mesh, method="bilinear")
    grid_input_2 = np.array([[0, 0], [1, 0], [4, 1]])
    face_mesh_output = grid_to_face_mesh_regridder.regrid(grid_input_2)
    expected_face_mesh_output = np.array([0.0, 1.4888258584989558])
    assert ma.allclose(expected_face_mesh_output, face_mesh_output)

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
