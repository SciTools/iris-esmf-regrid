"""Provides load/save functions for regridders."""

from contextlib import contextmanager

import iris
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
import numpy as np
import scipy.sparse

import esmf_regrid
from esmf_regrid import Constants, _load_context, check_method, esmpy
from esmf_regrid.experimental._partial import PartialRegridder
from esmf_regrid.experimental.unstructured_scheme import (
    GridToMeshESMFRegridder,
    MeshToGridESMFRegridder,
)
from esmf_regrid.schemes import (
    ESMFAreaWeighted,
    ESMFAreaWeightedRegridder,
    ESMFBilinear,
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
    PartialRegridder,
]
_REGRIDDER_NAME_MAP = {rg_class.__name__: rg_class for rg_class in SUPPORTED_REGRIDDERS}
_SOURCE_NAME = "regridder_source_field"
_SOURCE_MASK_NAME = "regridder_source_mask"
_TARGET_NAME = "regridder_target_field"
_TARGET_MASK_NAME = "regridder_target_mask"
_WEIGHTS_NAME = "regridder_weights"
_WEIGHTS_SHAPE_NAME = "weights_shape"
_WEIGHTS_ROW_NAME = "weight_matrix_rows"
_WEIGHTS_COL_NAME = "weight_matrix_columns"
_REGRIDDER_TYPE = "regridder_type"
_VERSION_ESMF = "ESMF_version"
_VERSION_INITIAL = "esmf_regrid_version_on_initialise"
_MDTOL = "mdtol"
_METHOD = "method"
_RESOLUTION = "resolution"
_SOURCE_RESOLUTION = "src_resolution"
_TARGET_RESOLUTION = "tgt_resolution"
_ESMF_ARGS = "esmf_args"
_SRC_SLICE_NAME = "src_slice"
_TGT_SLICE_NAME = "tgt_slice"
_VALID_ESMF_KWARGS = [
    "pole_method",
    "regrid_pole_npoints",
    "line_type",
    "extrap_method",
    "extrap_num_src_pnts",
    "extrap_dist_exponent",
    "extrap_num_levels",
    "unmapped_action",
    "ignore_degenerate",
    "large_file",
]
_POLE_METHOD_DICT = {e.name: e for e in esmpy.PoleMethod}
_LINE_TYPE_DICT = {e.name: e for e in esmpy.LineType}
_EXTRAP_METHOD_DICT = {e.name: e for e in esmpy.ExtrapMethod}
_UNMAPPED_ACTION_DICT = {e.name: e for e in esmpy.UnmappedAction}
_ESMF_ENUM_ARGS = {
    "pole_method": _POLE_METHOD_DICT,
    "line_type": _LINE_TYPE_DICT,
    "extrap_method": _EXTRAP_METHOD_DICT,
    "unmapped_action": _UNMAPPED_ACTION_DICT,
}
_ESMF_BOOL_ARGS = ["ignore_degenerate", "large_file"]


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
        src_coord_names = [coord.var_name for coord in src_mesh_coords]
    tgt_coord_names = []
    tgt_mesh_coords = []
    if tgt_cube.mesh is not None:
        tgt_mesh = tgt_cube.mesh
        tgt_mesh_coords = tgt_mesh.coords()
        tgt_coord_names = [coord.var_name for coord in tgt_mesh_coords]

    try:
        for coord in src_mesh_coords:
            coord.var_name = f"{_SOURCE_NAME}_mesh_{coord.name()}"
        for coord in tgt_mesh_coords:
            coord.var_name = f"{_TARGET_NAME}_mesh_{coord.name()}"
        yield None
    finally:
        for coord, var_name in zip(src_mesh_coords, src_coord_names, strict=False):
            coord.var_name = var_name
        for coord, var_name in zip(tgt_mesh_coords, tgt_coord_names, strict=False):
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


def _generate_src_tgt(regridder_type, rg, allow_partial):
    if regridder_type in [
        "ESMFAreaWeightedRegridder",
        "ESMFBilinearRegridder",
        "ESMFNearestRegridder",
        "PartialRegridder",
    ]:
        if regridder_type == "PartialRegridder" and not allow_partial:
            e_msg = "To save a PartialRegridder, `allow_partial=True` must be set."
            raise ValueError(e_msg)
        src_grid = rg._src
        if isinstance(src_grid, GridRecord):
            src_cube = _standard_grid_cube(
                (src_grid.grid_y, src_grid.grid_x), _SOURCE_NAME
            )
        elif isinstance(src_grid, MeshRecord):
            src_mesh, src_location = src_grid
            src_cube = _standard_mesh_cube(src_mesh, src_location, _SOURCE_NAME)
        else:
            e_msg = "Improper type for `rg._src`."
            raise ValueError(e_msg)
        _add_mask_to_cube(rg.src_mask, src_cube, _SOURCE_MASK_NAME)

        tgt_grid = rg._tgt
        if isinstance(tgt_grid, GridRecord):
            tgt_cube = _standard_grid_cube(
                (tgt_grid.grid_y, tgt_grid.grid_x), _TARGET_NAME
            )
        elif isinstance(tgt_grid, MeshRecord):
            tgt_mesh, tgt_location = tgt_grid
            tgt_cube = _standard_mesh_cube(tgt_mesh, tgt_location, _TARGET_NAME)
        else:
            e_msg = "Improper type for `rg._tgt`."
            raise ValueError(e_msg)
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, _TARGET_MASK_NAME)
    elif regridder_type == "GridToMeshESMFRegridder":
        src_grid = (rg.grid_y, rg.grid_x)
        src_cube = _standard_grid_cube(src_grid, _SOURCE_NAME)
        _add_mask_to_cube(rg.src_mask, src_cube, _SOURCE_MASK_NAME)

        tgt_mesh = rg.mesh
        tgt_location = rg.location
        tgt_cube = _standard_mesh_cube(tgt_mesh, tgt_location, _TARGET_NAME)
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, _TARGET_MASK_NAME)

    elif regridder_type == "MeshToGridESMFRegridder":
        src_mesh = rg.mesh
        src_location = rg.location
        src_cube = _standard_mesh_cube(src_mesh, src_location, _SOURCE_NAME)
        _add_mask_to_cube(rg.src_mask, src_cube, _SOURCE_MASK_NAME)

        tgt_grid = (rg.grid_y, rg.grid_x)
        tgt_cube = _standard_grid_cube(tgt_grid, _TARGET_NAME)
        _add_mask_to_cube(rg.tgt_mask, tgt_cube, _TARGET_MASK_NAME)

    else:
        e_msg = f"Unexpected regridder type {regridder_type}."
        raise TypeError(e_msg)
    return src_cube, tgt_cube


def save_regridder(rg, filename, allow_partial=False):
    """Save a regridder scheme instance.

    Saves any of the regridder classes, i.e.
    :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`,
    :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`,
    :class:`~esmf_regrid.schemes.ESMFAreaWeightedRegridder`,
    :class:`~esmf_regrid.schemes.ESMFBilinearRegridder` or
    :class:`~esmf_regrid.schemes.ESMFNearestRegridder`.
    .

    Parameters
    ----------
    rg : :class:`~esmf_regrid.schemes._ESMFRegridder`
        The regridder instance to save.
    filename : str
        The file name to save to.
    allow_partial : bool, default=False
        If True, allow the saving of :class:`~esmf_regrid.experimental._partial.PartialRegridder` instances.
    """
    regridder_type = rg.__class__.__name__

    src_cube, tgt_cube = _generate_src_tgt(regridder_type, rg, allow_partial)

    method = str(check_method(rg.method).name)

    if regridder_type in ["GridToMeshESMFRegridder", "MeshToGridESMFRegridder"]:
        resolution = rg.resolution
        src_resolution = None
        tgt_resolution = None
    elif method == "CONSERVATIVE":
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
        _REGRIDDER_TYPE: regridder_type,
        _VERSION_ESMF: esmf_version,
        _VERSION_INITIAL: esmf_regrid_version,
        "esmf_regrid_version_on_save": save_version,
        "normalization": normalization,
        _MDTOL: mdtol,
        _METHOD: method,
    }
    if resolution is not None:
        attributes[_RESOLUTION] = resolution
    if src_resolution is not None:
        attributes[_SOURCE_RESOLUTION] = src_resolution
    if tgt_resolution is not None:
        attributes[_TARGET_RESOLUTION] = tgt_resolution

    extra_cubes = []
    if regridder_type == "PartialRegridder":
        src_slice = rg.src_slice  # this slice is described by a tuple
        if src_slice is None:
            src_slice = []
        src_slice_cube = Cube(
            src_slice, long_name=_SRC_SLICE_NAME, var_name=_SRC_SLICE_NAME
        )
        tgt_slice = rg.tgt_slice  # this slice is described by a tuple
        if tgt_slice is None:
            tgt_slice = []
        tgt_slice_cube = Cube(
            tgt_slice, long_name=_TGT_SLICE_NAME, var_name=_TGT_SLICE_NAME
        )
        extra_cubes = [src_slice_cube, tgt_slice_cube]

    weights_cube = Cube(weight_data, var_name=_WEIGHTS_NAME, long_name=_WEIGHTS_NAME)
    row_coord = AuxCoord(
        weight_rows, var_name=_WEIGHTS_ROW_NAME, long_name=_WEIGHTS_ROW_NAME
    )
    col_coord = AuxCoord(
        weight_cols, var_name=_WEIGHTS_COL_NAME, long_name=_WEIGHTS_COL_NAME
    )
    weights_cube.add_aux_coord(row_coord, 0)
    weights_cube.add_aux_coord(col_coord, 0)

    esmf_args = rg.esmf_args
    if esmf_args is None:
        esmf_args = {}
    for arg in esmf_args:
        if arg not in _VALID_ESMF_KWARGS:
            e_msg = f"{arg} is not considered a valid argument to pass to ESMF."
            raise KeyError(e_msg)
    esmf_arg_attributes = {
        k: v.name if hasattr(v, "name") else int(v) if isinstance(v, bool) else v
        for k, v in esmf_args.items()
    }
    esmf_arg_coord = AuxCoord(
        0, var_name=_ESMF_ARGS, long_name=_ESMF_ARGS, attributes=esmf_arg_attributes
    )
    weights_cube.add_aux_coord(esmf_arg_coord)

    weight_shape_cube = Cube(
        weight_shape,
        var_name=_WEIGHTS_SHAPE_NAME,
        long_name=_WEIGHTS_SHAPE_NAME,
    )

    # Save cubes while ensuring var_names do not conflict for the sake of consistency.
    with _managed_var_name(src_cube, tgt_cube):
        cube_list = CubeList(
            [src_cube, tgt_cube, weights_cube, weight_shape_cube, *extra_cubes]
        )

        for cube in cube_list:
            cube.attributes = attributes

        iris.fileformats.netcdf.save(cube_list, filename)


def load_regridder(filename, allow_partial=False):
    """Load a regridder scheme instance.

    Loads any of the regridder classes, i.e.
    :class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`,
    :class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder`,
    :class:`~esmf_regrid.schemes.ESMFAreaWeightedRegridder`,
    :class:`~esmf_regrid.schemes.ESMFBilinearRegridder` or
    :class:`~esmf_regrid.schemes.ESMFNearestRegridder`.

    Parameters
    ----------
    filename : str
        The file name to load from.
    allow_partial : bool, default=False
        If True, allow the loading of :class:`~esmf_regrid.experimental._partial.PartialRegridder` instances.

    Returns
    -------
    :class:`~esmf_regrid.schemes._ESMFRegridder`
    """
    with _load_context():
        cubes = iris.load(filename)

    # Extract the source, target and metadata information.
    src_cube = cubes.extract_cube(_SOURCE_NAME)
    _clean_var_names(src_cube)
    tgt_cube = cubes.extract_cube(_TARGET_NAME)
    _clean_var_names(tgt_cube)
    weights_cube = cubes.extract_cube(_WEIGHTS_NAME)
    weight_shape_cube = cubes.extract_cube(_WEIGHTS_SHAPE_NAME)

    # Determine the regridder type.
    regridder_type = weights_cube.attributes[_REGRIDDER_TYPE]
    if regridder_type not in _REGRIDDER_NAME_MAP:
        e_msg = f"Unrecognised regridder type {regridder_type}."
        raise TypeError(e_msg)
    scheme = _REGRIDDER_NAME_MAP[regridder_type]

    if regridder_type == "PartialRegridder" and not allow_partial:
        e_msg = (
            "PartialRegridder cannot be loaded without setting `allow_partial=True`."
        )
        raise ValueError(e_msg)

    # Determine the regridding method, allowing for files created when
    # conservative regridding was the only method.
    method_string = weights_cube.attributes.get(_METHOD, "CONSERVATIVE")
    # Account for strings saved in previous versions.
    method_string = method_string.upper()
    method = getattr(Constants.Method, method_string)

    resolution = weights_cube.attributes.get(_RESOLUTION, None)
    src_resolution = weights_cube.attributes.get(_SOURCE_RESOLUTION, None)
    tgt_resolution = weights_cube.attributes.get(_TARGET_RESOLUTION, None)
    if resolution is not None:
        resolution = int(resolution)
    if src_resolution is not None:
        src_resolution = int(src_resolution)
    if tgt_resolution is not None:
        tgt_resolution = int(tgt_resolution)

    # Reconstruct the weight matrix.
    weight_data = weights_cube.data
    weight_rows = weights_cube.coord(_WEIGHTS_ROW_NAME).points
    weight_cols = weights_cube.coord(_WEIGHTS_COL_NAME).points
    weight_shape = weight_shape_cube.data
    weight_matrix = scipy.sparse.csr_matrix(
        (weight_data, (weight_rows, weight_cols)), shape=weight_shape
    )

    mdtol = weights_cube.attributes[_MDTOL]

    if src_cube.coords(_SOURCE_MASK_NAME):
        use_src_mask = src_cube.coord(_SOURCE_MASK_NAME).points.astype(bool)
    else:
        use_src_mask = False
    if tgt_cube.coords(_TARGET_MASK_NAME):
        use_tgt_mask = tgt_cube.coord(_TARGET_MASK_NAME).points.astype(bool)
    else:
        use_tgt_mask = False

    # Allow for this coord not to exist for the sake of backwards compatibility.
    esmf_args_coords = weights_cube.coords(_ESMF_ARGS)
    if len(esmf_args_coords) == 0:
        esmf_args = {}
    else:
        esmf_args = esmf_args_coords[0].attributes
    for arg, arg_dict in _ESMF_ENUM_ARGS.items():
        if arg in esmf_args:
            esmf_args[arg] = arg_dict[esmf_args[arg]]
    for arg in _ESMF_BOOL_ARGS:
        if arg in esmf_args:
            esmf_args[arg] = bool(esmf_args[arg])

    if scheme is GridToMeshESMFRegridder:
        resolution_keyword = _SOURCE_RESOLUTION
        kwargs = {resolution_keyword: resolution, "method": method, "mdtol": mdtol}
    elif scheme is MeshToGridESMFRegridder:
        resolution_keyword = _TARGET_RESOLUTION
        kwargs = {resolution_keyword: resolution, "method": method, "mdtol": mdtol}
    elif method is Constants.Method.CONSERVATIVE:
        kwargs = {
            _SOURCE_RESOLUTION: src_resolution,
            _TARGET_RESOLUTION: tgt_resolution,
            "mdtol": mdtol,
        }
    elif method is Constants.Method.BILINEAR:
        kwargs = {"mdtol": mdtol}
    else:
        kwargs = {}

    if scheme is PartialRegridder:
        src_slice = cubes.extract_cube(_SRC_SLICE_NAME).data.tolist()
        if src_slice == []:
            src_slice = None
        tgt_slice = cubes.extract_cube(_TGT_SLICE_NAME).data.tolist()
        if tgt_slice == []:
            tgt_slice = None
        sub_scheme = {
            Constants.Method.CONSERVATIVE: ESMFAreaWeighted,
            Constants.Method.BILINEAR: ESMFBilinear,
        }[method]
        mdtol = kwargs.pop(_MDTOL, None)
        sub_kwargs = {}
        if mdtol is not None:
            sub_kwargs[_MDTOL] = mdtol
        regridder = scheme(
            src_cube,
            tgt_cube,
            src_slice,
            tgt_slice,
            weight_matrix,
            sub_scheme(
                use_src_mask=use_src_mask,
                use_tgt_mask=use_tgt_mask,
                esmf_args=esmf_args,
                **sub_kwargs,
            ),
            **kwargs,
        )
    else:
        regridder = scheme(
            src_cube,
            tgt_cube,
            precomputed_weights=weight_matrix,
            use_src_mask=use_src_mask,
            use_tgt_mask=use_tgt_mask,
            esmf_args=esmf_args,
            **kwargs,
        )

    esmf_version = weights_cube.attributes[_VERSION_ESMF]
    regridder.regridder.esmf_version = esmf_version
    esmf_regrid_version = weights_cube.attributes[_VERSION_INITIAL]
    regridder.regridder.esmf_regrid_version = esmf_regrid_version
    return regridder
