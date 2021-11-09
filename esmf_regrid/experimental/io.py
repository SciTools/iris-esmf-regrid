"""Provides load/save functions for regridders."""

import iris
from iris.cube import Cube, CubeList
from iris.coords import AuxCoord
from esmf_regrid.experimental.unstructured_scheme import (
    MeshToGridESMFRegridder,
    GridToMeshESMFRegridder,
)


def save_regridder(rg, file):
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
        tgt_data = np.zeros(tgt_mesh.face_node_connectivity.indices.shape)
        tgt_cube = Cube(tgt_data, long_name=tgt_name)
        for coord in tgt_mesh.to_MeshCoords("face"):
            tgt_cube.add_aux_coord(coord, 0)
    elif isinstance(rg, MeshToGridESMFRegridder):
        regridder_type = "mesh to grid"
        src_mesh = rg.mesh
        src_data = np.zeros(src_mesh.face_node_connectivity.indices.shape)
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

    cube_list = CubeList([src_cube, tgt_cube, metadata_cube])
    iris.fileformats.netcdf.save(cube_list, file)
