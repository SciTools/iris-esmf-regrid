"""Provides load/save functions for regridders."""

import iris
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy as np
import scipy.sparse

from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)


def save_regridder(rg, filename):
    """
    Saves a regridder scheme instance.

    Saves either a `GridToMeshESMFRegridder` or a `MeshToGridESMFRegridder`.

    Parameters
    ----------
    rg : GridToMeshESMFRegridder, MeshToGridESMFRegridder
        The regridder instance to save.
    filename : str
        The file name to save to.
    """
    src_name = "regridder source field"
    tgt_name = "regridder target field"
    if isinstance(rg, GridToMeshESMFRegridder):
        regridder_type = "grid to mesh"
        src_grid = (rg.grid_y, rg.grid_x)
        src_shape = [len(coord.points) for coord in src_grid]
        src_data = np.zeros(src_shape)
        src_cube = Cube(src_data, long_name=src_name)
        src_cube.add_dim_coord(src_grid[0], 0)
        src_cube.add_dim_coord(src_grid[1], 1)

        tgt_mesh = rg.mesh
        tgt_data = np.zeros(tgt_mesh.face_node_connectivity.indices.shape[0])
        tgt_cube = Cube(tgt_data, long_name=tgt_name)
        for coord in tgt_mesh.to_MeshCoords("face"):
            tgt_cube.add_aux_coord(coord, 0)
    elif isinstance(rg, MeshToGridESMFRegridder):
        regridder_type = "mesh to grid"
        src_mesh = rg.mesh
        src_data = np.zeros(src_mesh.face_node_connectivity.indices.shape[0])
        src_cube = Cube(src_data, long_name=src_name)
        for coord in src_mesh.to_MeshCoords("face"):
            src_cube.add_aux_coord(coord, 0)

        tgt_grid = (rg.grid_y, rg.grid_x)
        tgt_shape = [len(coord.points) for coord in tgt_grid]
        tgt_data = np.zeros(tgt_shape)
        tgt_cube = Cube(tgt_data, long_name=tgt_name)
        tgt_cube.add_dim_coord(tgt_grid[0], 0)
        tgt_cube.add_dim_coord(tgt_grid[1], 1)
    else:
        msg = (
            f"Expected a regridder of type `GridToMeshESMFRegridder` or "
            f"`MeshToGridESMFRegridder`, got type {type(rg)}"
        )
        raise TypeError(msg)

    metadata_name = "regridder weights and metadata"

    weight_matrix = rg.regridder.weight_matrix
    reformatted_weight_matrix = weight_matrix.tocoo()
    weight_data = reformatted_weight_matrix.data
    weight_rows = reformatted_weight_matrix.row
    weight_cols = reformatted_weight_matrix.col
    weight_shape = reformatted_weight_matrix.shape

    mdtol = rg.mdtol
    attributes = {
        "regridder type": regridder_type,
        "mdtol": mdtol,
        "weights shape": weight_shape,
    }

    metadata_cube = Cube(weight_data, long_name=metadata_name, attributes=attributes)
    row_name = "weight matrix rows"
    row_coord = AuxCoord(weight_rows, long_name=row_name)
    col_name = "weight matrix columns"
    col_coord = AuxCoord(weight_cols, long_name=col_name)
    metadata_cube.add_aux_coord(row_coord, 0)
    metadata_cube.add_aux_coord(col_coord, 0)

    # Avoid saving bug.
    if regridder_type == "grid to mesh":
        cube_list = CubeList([src_cube, tgt_cube, metadata_cube])
    elif regridder_type == "mesh to grid":
        cube_list = CubeList([tgt_cube, src_cube, metadata_cube])
    iris.fileformats.netcdf.save(cube_list, filename)


def load_regridder(filename):
    """
    Loads a regridder scheme instance.

    Loads either a `GridToMeshESMFRegridder` or a `MeshToGridESMFRegridder`.

    Parameters
    ----------
    filename : str
        The file name to load from.
    """
    with PARSE_UGRID_ON_LOAD.context():
        cubes = iris.load(filename)

    src_name = "regridder source field"
    tgt_name = "regridder target field"
    metadata_name = "regridder weights and metadata"

    src_cube = cubes.extract_cube(src_name)
    tgt_cube = cubes.extract_cube(tgt_name)
    metadata_cube = cubes.extract_cube(metadata_name)

    regridder_type = metadata_cube.attributes["regridder type"]
    if regridder_type == "grid to mesh":
        scheme = GridToMeshESMFRegridder
    elif regridder_type == "mesh to grid":
        scheme = MeshToGridESMFRegridder
    else:
        raise ValueError(
            f"expected a regridder type to be either 'grid to mesh' "
            f"or 'mesh to grid', got '{regridder_type}'"
        )
    weight_data = metadata_cube.data
    row_name = "weight matrix rows"
    weight_rows = metadata_cube.coord(row_name).points
    col_name = "weight matrix columns"
    weight_cols = metadata_cube.coord(col_name).points
    weight_shape = metadata_cube.attributes["weights shape"]
    weight_matrix = scipy.sparse.csr_matrix(
        (weight_data, (weight_rows, weight_cols)), shape=weight_shape
    )
    mdtol = metadata_cube.attributes["mdtol"]

    regridder = scheme(
        src_cube, tgt_cube, mdtol=mdtol, precomputed_weights=weight_matrix
    )
    return regridder
