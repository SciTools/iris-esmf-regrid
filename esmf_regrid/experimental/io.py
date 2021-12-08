"""Provides load/save functions for regridders."""

import iris
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy as np
import scipy.sparse

import esmf_regrid
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)


SUPPORTED_REGRIDDERS = [
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
]
REGRIDDER_NAME_MAP = {rg_class.__name__: rg_class for rg_class in SUPPORTED_REGRIDDERS}


def save_regridder(rg, filename):
    """
    Save a regridder scheme instance.

    Saves either a `GridToMeshESMFRegridder` or a `MeshToGridESMFRegridder`.

    Parameters
    ----------
    rg : GridToMeshESMFRegridder, MeshToGridESMFRegridder
        The regridder instance to save.
    filename : str
        The file name to save to.
    """
    src_name = "regridder_source_field"
    tgt_name = "regridder_target_field"
    regridder_type = rg.__class__.__name__
    if regridder_type == "GridToMeshESMFRegridder":
        src_grid = (rg.grid_y, rg.grid_x)
        src_shape = [len(coord.points) for coord in src_grid]
        src_data = np.zeros(src_shape)
        src_cube = Cube(src_data, var_name=src_name, long_name=src_name)
        src_cube.add_dim_coord(src_grid[0], 0)
        src_cube.add_dim_coord(src_grid[1], 1)

        tgt_mesh = rg.mesh
        tgt_data = np.zeros(tgt_mesh.face_node_connectivity.indices.shape[0])
        tgt_cube = Cube(tgt_data, var_name=tgt_name, long_name=tgt_name)
        for coord in tgt_mesh.to_MeshCoords("face"):
            tgt_cube.add_aux_coord(coord, 0)
    elif regridder_type == "MeshToGridESMFRegridder":
        src_mesh = rg.mesh
        src_data = np.zeros(src_mesh.face_node_connectivity.indices.shape[0])
        src_cube = Cube(src_data, var_name=src_name, long_name=src_name)
        for coord in src_mesh.to_MeshCoords("face"):
            src_cube.add_aux_coord(coord, 0)

        tgt_grid = (rg.grid_y, rg.grid_x)
        tgt_shape = [len(coord.points) for coord in tgt_grid]
        tgt_data = np.zeros(tgt_shape)
        tgt_cube = Cube(tgt_data, var_name=tgt_name, long_name=tgt_name)
        tgt_cube.add_dim_coord(tgt_grid[0], 0)
        tgt_cube.add_dim_coord(tgt_grid[1], 1)
    else:
        msg = (
            f"Expected a regridder of type `GridToMeshESMFRegridder` or "
            f"`MeshToGridESMFRegridder`, got type {regridder_type}."
        )
        raise TypeError(msg)

    weights_name = "regridder_weights"

    weight_matrix = rg.regridder.weight_matrix
    reformatted_weight_matrix = weight_matrix.tocoo()
    weight_data = reformatted_weight_matrix.data
    weight_rows = reformatted_weight_matrix.row
    weight_cols = reformatted_weight_matrix.col
    weight_shape = reformatted_weight_matrix.shape

    esmf_version = rg.regridder.esmf_version
    # if esmf_version is None:
    #     esmf_version = "unknown"
    esmf_regrid_version = rg.regridder.esmf_regrid_version
    save_version = esmf_regrid.__version__

    mdtol = rg.mdtol
    attributes = {
        "title": "iris-esmf-regrid regridding scheme",
        "regridder_type": regridder_type,
        "ESMF_version": esmf_version,
        "esmf_regrid_version_on_initialise": esmf_regrid_version,
        "esmf_regrid_version_on_save": save_version,
        "mdtol": mdtol,
    }

    weights_cube = Cube(weight_data, var_name=weights_name, long_name=weights_name)
    row_name = "weight_matrix_rows"
    row_coord = AuxCoord(weight_rows, var_name=row_name, long_name=row_name)
    col_name = "weight_matrix_columns"
    col_coord = AuxCoord(weight_cols, var_name=col_name, long_name=col_name)
    weights_cube.add_aux_coord(row_coord, 0)
    weights_cube.add_aux_coord(col_coord, 0)

    weight_shape_cube = Cube(
        weight_shape,
        var_name="weights_shape",
        long_name="weights_shape",
    )

    # Avoid saving bug by placing the mesh cube second.
    # TODO: simplify this when this bug is fixed in iris.
    if regridder_type == "GridToMeshESMFRegridder":
        cube_list = CubeList([src_cube, tgt_cube, weights_cube, weight_shape_cube])
    elif regridder_type == "MeshToGridESMFRegridder":
        cube_list = CubeList([tgt_cube, src_cube, weights_cube, weight_shape_cube])

    for cube in cube_list:
        cube.attributes = attributes

    iris.fileformats.netcdf.save(cube_list, filename)


def load_regridder(filename):
    """
    Load a regridder scheme instance.

    Loads either a `GridToMeshESMFRegridder` or a `MeshToGridESMFRegridder`.

    Parameters
    ----------
    filename : str
        The file name to load from.
    """
    with PARSE_UGRID_ON_LOAD.context():
        cubes = iris.load(filename)

    src_name = "regridder_source_field"
    tgt_name = "regridder_target_field"
    weights_name = "regridder_weights"

    # Extract the source, target and metadata information.
    src_cube = cubes.extract_cube(src_name)
    tgt_cube = cubes.extract_cube(tgt_name)
    weights_cube = cubes.extract_cube(weights_name)
    weight_shape_cube = cubes.extract_cube("weights_shape")

    # Determine the regridder type.
    regridder_type = weights_cube.attributes["regridder_type"]
    assert regridder_type in REGRIDDER_NAME_MAP.keys()
    scheme = REGRIDDER_NAME_MAP[regridder_type]

    # Reconstruct the weight matrix.
    weight_data = weights_cube.data
    row_name = "weight_matrix_rows"
    weight_rows = weights_cube.coord(row_name).points
    col_name = "weight_matrix_columns"
    weight_cols = weights_cube.coord(col_name).points
    weight_shape = weight_shape_cube.data
    weight_matrix = scipy.sparse.csr_matrix(
        (weight_data, (weight_rows, weight_cols)), shape=weight_shape
    )

    mdtol = weights_cube.attributes["mdtol"]

    regridder = scheme(
        src_cube, tgt_cube, mdtol=mdtol, precomputed_weights=weight_matrix
    )

    esmf_version = weights_cube.attributes["ESMF_version"]
    regridder.regridder.esmf_version = esmf_version
    esmf_regrid_version = weights_cube.attributes["esmf_regrid_version_on_initialise"]
    regridder.regridder.esmf_regrid_version = esmf_regrid_version
    return regridder
