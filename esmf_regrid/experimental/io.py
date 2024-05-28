"""Provides load/save functions for regridders."""

from contextlib import contextmanager

import iris
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy as np
import scipy.sparse

import esmf_regrid
from esmf_regrid import check_method, Constants
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import (
    ESMFAreaWeightedRegridder,
    ESMFBilinearRegridder,
    ESMFNearestRegridder,
    GridRecord,
    MeshRecord,
)


SUPPORTED_REGRIDDERS = [
    ESMFAreaWeightedRegridder,
    ESMFBilinearRegridder,
    ESMFNearestRegridder,
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
]
REGRIDDER_NAME_MAP = {rg_class.__name__: rg_class for rg_class in SUPPORTED_REGRIDDERS}
SOURCE_NAME = "regridder_source_field"
SOURCE_MASK_NAME = "regridder_source_mask"
TARGET_NAME = "regridder_target_field"
TARGET_MASK_NAME = "regridder_target_mask"
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
SOURCE_RESOLUTION = "src_resolution"
TARGET_RESOLUTION = "tgt_resolution"


def _add_mask_to_cube(mask, cube, name):
    if isinstance(mask, np.ndarray):
        mask = mask.astype(int)
        mask_coord = AuxCoord(mask, var_name=name, long_name=name)
        cube.add_aux_coord(mask_coord, list(range(cube.ndim)))


@contextmanager
def _managed_var_name(src_cube, tgt_cube):
    src_coord_names = []
    src_mesh_coords = []
    if src_cube.mesh is not None:
        src_mesh = src_cube.mesh
        src_mesh_coords = src_mesh.coords()
        for coord in src_mesh_coords:
            src_coord_names.append(coord.var_name)
    tgt_coord_names = []
    tgt_mesh_coords = []
    if tgt_cube.mesh is not None:
        tgt_mesh = tgt_cube.mesh
        tgt_mesh_coords = tgt_mesh.coords()
        for coord in tgt_mesh_coords:
            tgt_coord_names.append(coord.var_name)

    try:
        for coord in src_mesh_coords:
            coord.var_name = "_".join([SOURCE_NAME, "mesh", coord.name()])
        for coord in tgt_mesh_coords:
            coord.var_name = "_".join([TARGET_NAME, "mesh", coord.name()])
        yield None
    finally:
        for coord, var_name in zip(src_mesh_coords, src_coord_names):
            coord.var_name = var_name
        for coord, var_name in zip(tgt_mesh_coords, tgt_coord_names):
            coord.var_name = var_name


def _clean_var_names(cube):
    cube.var_name = None
    for coord in cube.coords():
        coord.var_name = None
    if cube.mesh is not None:
        cube.mesh.var_name = None
        for coord in cube.mesh.coords():
            coord.var_name = None
        for con in cube.mesh.connectivities():
            con.var_name = None
    return cube


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

    def _standard_mesh_cube(mesh, location, name):
        mesh_coords = mesh.to_MeshCoords(location)
        data = np.zeros(mesh_coords[0].points.shape[0])
        cube = Cube(data, var_name=name, long_name=name)
        for coord in mesh_coords:
            cube.add_aux_coord(coord, 0)
        return cube

    if regridder_type in [
        "ESMFAreaWeightedRegridder",
        "ESMFBilinearRegridder",
        "ESMFNearestRegridder",
    ]:
        src_grid = rg._src
        if isinstance(src_grid, GridRecord):
            src_cube = _standard_grid_cube(
                (src_grid.grid_y, src_grid.grid_x), SOURCE_NAME
            )
        elif isinstance(src_grid, MeshRecord):
            src_mesh, src_location = src_grid
            src_cube = _standard_mesh_cube(src_mesh, src_location, SOURCE_NAME)
        else:
            raise ValueError("Improper type for `rg._src`.")
        _add_mask_to_cube(rg.src_mask, src_cube, SOURCE_MASK_NAME)

        tgt_grid = rg._tgt
        if isinstance(tgt_grid, GridRecord):
            tgt_cube = _standard_grid_cube(
                (tgt_grid.grid_y, tgt_grid.grid_x), TARGET_NAME
            )
        elif isinstance(tgt_grid, MeshRecord):
            tgt_mesh, tgt_location = tgt_grid
            tgt_cube = _standard_mesh_cube(tgt_mesh, tgt_location, TARGET_NAME)
        else:
            raise ValueError("Improper type for `rg._tgt`.")
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, TARGET_MASK_NAME)
    elif regridder_type == "GridToMeshESMFRegridder":
        src_grid = (rg.grid_y, rg.grid_x)
        src_cube = _standard_grid_cube(src_grid, SOURCE_NAME)
        _add_mask_to_cube(rg.src_mask, src_cube, SOURCE_MASK_NAME)

        tgt_mesh = rg.mesh
        tgt_location = rg.location
        tgt_cube = _standard_mesh_cube(tgt_mesh, tgt_location, TARGET_NAME)
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, TARGET_MASK_NAME)

    elif regridder_type == "MeshToGridESMFRegridder":
        src_mesh = rg.mesh
        src_location = rg.location
        src_cube = _standard_mesh_cube(src_mesh, src_location, SOURCE_NAME)
        _add_mask_to_cube(rg.src_mask, src_cube, SOURCE_MASK_NAME)

        tgt_grid = (rg.grid_y, rg.grid_x)
        tgt_cube = _standard_grid_cube(tgt_grid, TARGET_NAME)
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, TARGET_MASK_NAME)
    else:
        msg = (
            f"Expected a regridder of type `GridToMeshESMFRegridder` or "
            f"`MeshToGridESMFRegridder`, got type {regridder_type}."
        )
        raise TypeError(msg)

    method = str(check_method(rg.method).name)

    if regridder_type in ["GridToMeshESMFRegridder", "MeshToGridESMFRegridder"]:
        resolution = rg.resolution
        src_resolution = None
        tgt_resolution = None
    elif regridder_type == "ESMFAreaWeightedRegridder":
        resolution = None
        src_resolution = rg.src_resolution
        tgt_resolution = rg.tgt_resolution
    else:
        resolution = None
        src_resolution = None
        tgt_resolution = None

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
    if src_resolution is not None:
        attributes[SOURCE_RESOLUTION] = src_resolution
    if tgt_resolution is not None:
        attributes[TARGET_RESOLUTION] = tgt_resolution

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

    # Save cubes while ensuring var_names do not conflict for the sake of consistency.
    with _managed_var_name(src_cube, tgt_cube):
        cube_list = CubeList([src_cube, tgt_cube, weights_cube, weight_shape_cube])

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
    src_cube = _clean_var_names(cubes.extract_cube(SOURCE_NAME))
    tgt_cube = _clean_var_names(cubes.extract_cube(TARGET_NAME))
    weights_cube = cubes.extract_cube(WEIGHTS_NAME)
    weight_shape_cube = cubes.extract_cube(WEIGHTS_SHAPE_NAME)

    # Determine the regridder type.
    regridder_type = weights_cube.attributes[REGRIDDER_TYPE]
    assert regridder_type in REGRIDDER_NAME_MAP.keys()
    scheme = REGRIDDER_NAME_MAP[regridder_type]

    # Determine the regridding method, allowing for files created when
    # conservative regridding was the only method.
    method = getattr(
        Constants.Method, weights_cube.attributes.get(METHOD, "CONSERVATIVE")
    )

    resolution = weights_cube.attributes.get(RESOLUTION, None)
    src_resolution = weights_cube.attributes.get(SOURCE_RESOLUTION, None)
    tgt_resolution = weights_cube.attributes.get(TARGET_RESOLUTION, None)
    if resolution is not None:
        resolution = int(resolution)
    if src_resolution is not None:
        src_resolution = int(src_resolution)
    if tgt_resolution is not None:
        tgt_resolution = int(tgt_resolution)

    # Reconstruct the weight matrix.
    weight_data = weights_cube.data
    weight_rows = weights_cube.coord(WEIGHTS_ROW_NAME).points
    weight_cols = weights_cube.coord(WEIGHTS_COL_NAME).points
    weight_shape = weight_shape_cube.data
    weight_matrix = scipy.sparse.csr_matrix(
        (weight_data, (weight_rows, weight_cols)), shape=weight_shape
    )

    mdtol = weights_cube.attributes[MDTOL]

    if src_cube.coords(SOURCE_MASK_NAME):
        use_src_mask = src_cube.coord(SOURCE_MASK_NAME).points
    else:
        use_src_mask = False
    if tgt_cube.coords(TARGET_MASK_NAME):
        use_tgt_mask = tgt_cube.coord(TARGET_MASK_NAME).points
    else:
        use_tgt_mask = False

    if scheme is GridToMeshESMFRegridder:
        resolution_keyword = SOURCE_RESOLUTION
        kwargs = {resolution_keyword: resolution, "method": method, "mdtol": mdtol}
    elif scheme is MeshToGridESMFRegridder:
        resolution_keyword = TARGET_RESOLUTION
        kwargs = {resolution_keyword: resolution, "method": method, "mdtol": mdtol}
    elif scheme is ESMFAreaWeightedRegridder:
        kwargs = {
            SOURCE_RESOLUTION: src_resolution,
            TARGET_RESOLUTION: tgt_resolution,
            "mdtol": mdtol,
        }
    elif scheme is ESMFBilinearRegridder:
        kwargs = {"mdtol": mdtol}
    else:
        kwargs = {}

    regridder = scheme(
        src_cube,
        tgt_cube,
        precomputed_weights=weight_matrix,
        use_src_mask=use_src_mask,
        use_tgt_mask=use_tgt_mask,
        **kwargs,
    )

    esmf_version = weights_cube.attributes[VERSION_ESMF]
    regridder.regridder.esmf_version = esmf_version
    esmf_regrid_version = weights_cube.attributes[VERSION_INITIAL]
    regridder.regridder.esmf_regrid_version = esmf_regrid_version
    return regridder
