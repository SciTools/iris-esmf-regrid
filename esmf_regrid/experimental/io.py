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
SOURCE_NAME = "regridder_source_field"
TARGET_NAME = "regridder_target_field"
WEIGHTS_NAME = "regridder_weights"
WEIGHTS_SHAPE_NAME = "weights_shape"
WEIGHTS_ROW_NAME = "weight_matrix_rows"
WEIGHTS_COL_NAME = "weight_matrix_columns"
REGRIDDER_TYPE = "regridder_type"
VERSION_ESMF = "ESMF_version"
VERSION_INITIAL = "esmf_regrid_version_on_initialise"
MDTOL = "mdtol"
METHOD = "method"
RESOLUTION = "resolution"


def save_regridder(rg, filename):
    """
    Save a regridder scheme instance.

    Saves either a
    :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`
    or a
    :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Parameters
    ----------
    rg : :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder` or :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`
        The regridder instance to save.
    filename : str
        The file name to save to.
    """
    regridder_type = rg.__class__.__name__

    def _standard_grid_cube(grid, name):
        if grid[0].ndim == 1:
            shape = [coord.points.size for coord in grid]
        else:
            shape = grid[0].shape
        data = np.zeros(shape)
        cube = Cube(data, var_name=name, long_name=name)
        if grid[0].ndim == 1:
            cube.add_dim_coord(grid[0], 0)
            cube.add_dim_coord(grid[1], 1)
        else:
            cube.add_aux_coord(grid[0], [0, 1])
            cube.add_aux_coord(grid[1], [0, 1])
        return cube

    if regridder_type == "GridToMeshESMFRegridder":
        src_grid = (rg.grid_y, rg.grid_x)
        src_cube = _standard_grid_cube(src_grid, SOURCE_NAME)

        tgt_mesh = rg.mesh
        tgt_location = rg.location
        tgt_mesh_coords = tgt_mesh.to_MeshCoords(tgt_location)
        tgt_data = np.zeros(tgt_mesh_coords[0].points.shape[0])
        tgt_cube = Cube(tgt_data, var_name=TARGET_NAME, long_name=TARGET_NAME)
        for coord in tgt_mesh_coords:
            tgt_cube.add_aux_coord(coord, 0)
    elif regridder_type == "MeshToGridESMFRegridder":
        src_mesh = rg.mesh
        src_location = rg.location
        src_mesh_coords = src_mesh.to_MeshCoords(src_location)
        src_data = np.zeros(src_mesh_coords[0].points.shape[0])
        src_cube = Cube(src_data, var_name=SOURCE_NAME, long_name=SOURCE_NAME)
        for coord in src_mesh_coords:
            src_cube.add_aux_coord(coord, 0)

        tgt_grid = (rg.grid_y, rg.grid_x)
        tgt_cube = _standard_grid_cube(tgt_grid, TARGET_NAME)
    else:
        msg = (
            f"Expected a regridder of type `GridToMeshESMFRegridder` or "
            f"`MeshToGridESMFRegridder`, got type {regridder_type}."
        )
        raise TypeError(msg)

    method = rg.method
    resolution = rg.resolution

    weight_matrix = rg.regridder.weight_matrix
    reformatted_weight_matrix = scipy.sparse.coo_matrix(weight_matrix)
    weight_data = reformatted_weight_matrix.data
    weight_rows = reformatted_weight_matrix.row
    weight_cols = reformatted_weight_matrix.col
    weight_shape = reformatted_weight_matrix.shape

    esmf_version = rg.regridder.esmf_version
    esmf_regrid_version = rg.regridder.esmf_regrid_version
    save_version = esmf_regrid.__version__

    # Currently, all schemes use the fracarea normalization.
    normalization = "fracarea"

    mdtol = rg.mdtol
    attributes = {
        "title": "iris-esmf-regrid regridding scheme",
        REGRIDDER_TYPE: regridder_type,
        VERSION_ESMF: esmf_version,
        VERSION_INITIAL: esmf_regrid_version,
        "esmf_regrid_version_on_save": save_version,
        "normalization": normalization,
        MDTOL: mdtol,
        METHOD: method,
    }
    if resolution is not None:
        attributes[RESOLUTION] = resolution

    weights_cube = Cube(weight_data, var_name=WEIGHTS_NAME, long_name=WEIGHTS_NAME)
    row_coord = AuxCoord(
        weight_rows, var_name=WEIGHTS_ROW_NAME, long_name=WEIGHTS_ROW_NAME
    )
    col_coord = AuxCoord(
        weight_cols, var_name=WEIGHTS_COL_NAME, long_name=WEIGHTS_COL_NAME
    )
    weights_cube.add_aux_coord(row_coord, 0)
    weights_cube.add_aux_coord(col_coord, 0)

    weight_shape_cube = Cube(
        weight_shape,
        var_name=WEIGHTS_SHAPE_NAME,
        long_name=WEIGHTS_SHAPE_NAME,
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

    Loads either a
    :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`
    or a
    :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`.

    Parameters
    ----------
    filename : str
        The file name to load from.

    Returns
    -------
    :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder` or :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`
    """
    with PARSE_UGRID_ON_LOAD.context():
        cubes = iris.load(filename)

    # Extract the source, target and metadata information.
    src_cube = cubes.extract_cube(SOURCE_NAME)
    tgt_cube = cubes.extract_cube(TARGET_NAME)
    weights_cube = cubes.extract_cube(WEIGHTS_NAME)
    weight_shape_cube = cubes.extract_cube(WEIGHTS_SHAPE_NAME)

    # Determine the regridder type.
    regridder_type = weights_cube.attributes[REGRIDDER_TYPE]
    assert regridder_type in REGRIDDER_NAME_MAP.keys()
    scheme = REGRIDDER_NAME_MAP[regridder_type]

    # Determine the regridding method, allowing for files created when
    # conservative regridding was the only method.
    method = weights_cube.attributes.get(METHOD, "conservative")
    resolution = weights_cube.attributes.get(RESOLUTION, None)
    if resolution is not None:
        resolution = int(resolution)

    # Reconstruct the weight matrix.
    weight_data = weights_cube.data
    weight_rows = weights_cube.coord(WEIGHTS_ROW_NAME).points
    weight_cols = weights_cube.coord(WEIGHTS_COL_NAME).points
    weight_shape = weight_shape_cube.data
    weight_matrix = scipy.sparse.csr_matrix(
        (weight_data, (weight_rows, weight_cols)), shape=weight_shape
    )

    mdtol = weights_cube.attributes[MDTOL]

    regridder = scheme(
        src_cube,
        tgt_cube,
        mdtol=mdtol,
        method=method,
        precomputed_weights=weight_matrix,
        resolution=resolution,
    )

    esmf_version = weights_cube.attributes[VERSION_ESMF]
    regridder.regridder.esmf_version = esmf_version
    esmf_regrid_version = weights_cube.attributes[VERSION_INITIAL]
    regridder.regridder.esmf_regrid_version = esmf_regrid_version
    return regridder
