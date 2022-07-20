"""Provides an iris interface for regridding."""

from collections import namedtuple
import copy
import functools

from cf_units import Unit
import dask.array as da
from iris._lazy_data import map_complete_blocks
import iris.coords
import iris.cube
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo, RefinedGridInfo, Regridder
from esmf_regrid.experimental.unstructured_regrid import MeshInfo

__all__ = [
    "ESMFAreaWeighted",
    "ESMFAreaWeightedRegridder",
    "regrid_rectilinear_to_rectilinear",
]


def _get_coord(cube, axis):
    try:
        coord = cube.coord(axis=axis, dim_coords=True)
    except CoordinateNotFoundError:
        coord = cube.coord(axis=axis)
    return coord


def _get_mask(cube, use_mask=True):
    if use_mask is False:
        result = None
    elif use_mask is True:
        src_x, src_y = (_get_coord(cube, "x"), _get_coord(cube, "y"))

        horizontal_dims = set(cube.coord_dims(src_x)) | set(cube.coord_dims(src_y))
        other_dims = tuple(set(range(cube.ndim)) - horizontal_dims)

        # Find a representative slice of data that spans both horizontal coords.
        if cube.coord_dims(src_x) == cube.coord_dims(src_y):
            slices = cube.slices([src_x])
        else:
            slices = cube.slices([src_x, src_y])
        data = next(slices).data
        if np.ma.is_masked(data):
            # Check that the mask is constant along all other dimensions.
            full_mask = da.ma.getmaskarray(cube.core_data())
            if not np.array_equal(
                da.all(full_mask, axis=other_dims).compute(),
                da.any(full_mask, axis=other_dims).compute(),
            ):
                raise ValueError(
                    "The mask derived from the cube is not constant over non-horizontal dimensions."
                    "Consider passing in an explicit mask instead."
                )
            mask = np.ma.getmaskarray(data)
            # Due to structural reasons, the mask should be transposed for curvilinear grids.
            if cube.coord_dims(src_x) != cube.coord_dims(src_y):
                mask = mask.T
        else:
            mask = None
        result = mask
    else:
        result = use_mask
    return result


def _contiguous_masked(bounds, mask):
    """
    Return the (N+1, M+1) vertices for 2D coordinate bounds of shape (N, M, 4).

    Assumes the bounds are contiguous outside of the mask. As long as the only
    discontiguities are associated with masked points, the returned vertices will
    represent every bound associated with an unmasked point. This function is
    designed to replicate the behaviour of iris.Coord.contiguous_bounds() for such
    bounds. For each vertex in the returned array there are up to four choices of
    bounds to derive from. Bounds associated with umasked points will be prioritised
    in this choice.

    For example, suppose we have a masked array:

    # 0 0 0 0
    # 0 - - 0
    # 0 - - 0
    # 0 0 0 0

    Now the indices of the final bounds dimension correspond to positions on the
    vertex array. For a bound whose first indixes are (i, j) the corresponding
    position in the vertex array of the four final indices are as follows:
    0=(j, i), 1=(j, i+1), 2=(j+1, i+1), 3=(j+1, i)
    The indices of the bounds which the final array will derive from are as follows:

    # (0, 0, 0) (0, 1, 0) (0, 2, 0) (0, 3, 0) (0, 3, 1)
    # (1, 0, 0) (1, 0, 1) (0, 2, 3) (1, 3, 0) (1, 3, 1)
    # (2, 0, 0) (2, 0, 1) (1, 1, 2) (2, 3, 0) (2, 3, 1)
    # (3, 0, 0) (3, 1, 0) (3, 2, 0) (3, 3, 0) (3, 3, 1)
    # (3, 0, 3) (3, 1, 3) (3, 3, 3) (3, 3, 3) (3, 3, 2)

    Note that only the center bound derives from a masked cell as there are no
    unmasked points to derive from.
    """
    mask = np.array(mask, dtype=int)

    # Construct a blank template array in the shape of the output.
    blank_template = np.zeros([size + 1 for size in mask.shape], dtype=int)

    # Define the slices of the resulting vertex array which derive from the
    # bounds in index 0 to 3.
    slice_0 = np.s_[:-1, :-1]
    slice_1 = np.s_[:-1, 1:]
    slice_2 = np.s_[1:, 1:]
    slice_3 = np.s_[1:, :-1]

    # Define the bounds which will derive from the bounds in index 0.
    filter_0 = blank_template.copy()
    filter_0[slice_0] = 1 - mask
    # Ensure the corner points are covered appropriately.
    filter_0[0, 0] = 1

    # Define the bounds which will derive from the bounds in index 1.
    filter_1 = blank_template.copy()
    filter_1[slice_1] = 1 - mask
    # Ensure the corner and edge bounds are covered appropriately.
    filter_1[0, 1:] = 1
    # Do not cover any points already covered.
    filter_1 = filter_1 * (1 - filter_0)

    # Define the bounds which will derive from the bounds in index 3.
    filter_3 = blank_template.copy()
    filter_3[slice_3] = 1 - mask
    # Ensure the corner and edge bounds are covered appropriately.
    filter_3[1:, 0] = 1
    # Do not cover any points already covered.
    filter_3 = filter_3 * (1 - filter_0 - filter_1)

    # The bounds deriving from index 2 will be all those not already covered.
    filter_2 = 1 - filter_0 - filter_1 - filter_3

    # Copy the bounds information over into the appropriate places.
    contiguous_bounds = blank_template.astype(bounds.dtype)
    contiguous_bounds[slice_0] += bounds[:, :, 0] * filter_0[slice_0]
    contiguous_bounds[slice_1] += bounds[:, :, 1] * filter_1[slice_1]
    contiguous_bounds[slice_2] += bounds[:, :, 2] * filter_2[slice_2]
    contiguous_bounds[slice_3] += bounds[:, :, 3] * filter_3[slice_3]
    return contiguous_bounds


def _cube_to_GridInfo(cube, center=False, resolution=None, mask=None):
    # This is a simplified version of an equivalent function/method in PR #26.
    # It is anticipated that this function will be replaced by the one in PR #26.
    #
    # Returns a GridInfo object describing the horizontal grid of the cube.
    # This may be inherited from code written for the rectilinear regridding scheme.
    lon, lat = _get_coord(cube, "x"), _get_coord(cube, "y")
    #  Checks may cover units, coord systems (e.g. rotated pole), contiguous bounds.
    if cube.coord_system() is None:
        crs = None
    else:
        crs = cube.coord_system().as_cartopy_crs()
    londim, latdim = len(lon.shape), len(lat.shape)
    assert londim == latdim
    assert londim in (1, 2)
    if londim == 1:
        # Ensure coords come from a proper grid.
        assert isinstance(lon, iris.coords.DimCoord)
        assert isinstance(lat, iris.coords.DimCoord)
        circular = lon.circular
        lon_bound_array = lon.contiguous_bounds()
        lat_bound_array = lat.contiguous_bounds()
        # TODO: perform checks on lat/lon.
    elif londim == 2:
        assert cube.coord_dims(lon) == cube.coord_dims(lat)
        if not np.any(mask):
            assert lon.is_contiguous()
            assert lat.is_contiguous()
            lon_bound_array = lon.contiguous_bounds()
            lat_bound_array = lat.contiguous_bounds()
        else:
            lon_bound_array = _contiguous_masked(lon.bounds, mask)
            lat_bound_array = _contiguous_masked(lat.bounds, mask)
        # 2D coords must be AuxCoords, which do not have a circular attribute.
        circular = False
    lon_bound_array = lon.units.convert(lon_bound_array, Unit("degrees"))
    lat_bound_array = lat.units.convert(lat_bound_array, Unit("degrees"))
    if resolution is None:
        grid_info = GridInfo(
            lon.points,
            lat.points,
            lon_bound_array,
            lat_bound_array,
            crs=crs,
            circular=circular,
            center=center,
            mask=mask,
        )
    else:
        grid_info = RefinedGridInfo(
            lon_bound_array,
            lat_bound_array,
            crs=crs,
            resolution=resolution,
            mask=mask,
        )
    return grid_info


def _mesh_to_MeshInfo(mesh, location):
    # Returns a MeshInfo object describing the mesh of the cube.
    assert mesh.topology_dimension == 2
    if None in mesh.face_coords:
        elem_coords = None
    else:
        elem_coords = np.stack([coord.points for coord in mesh.face_coords], axis=-1)
    meshinfo = MeshInfo(
        np.stack([coord.points for coord in mesh.node_coords], axis=-1),
        mesh.face_node_connectivity.indices_by_location(),
        mesh.face_node_connectivity.start_index,
        elem_coords=elem_coords,
        location=location,
    )
    return meshinfo


def _regrid_along_grid_dims(regridder, data, grid_x_dim, grid_y_dim, mdtol):
    """
    Perform regridding on data over specific dimensions.

    Parameters
    ----------
    regridder : Regridder
        An instance of Regridder initialised to perfomr regridding.
    data : array
        The data to be regrid.
    grid_x_dim : int
        The dimension of the X axis.
    grid_y_dim : int
        The dimension of the Y axis.
    mdtol : float
        Tolerance of missing data.

    Returns
    -------
    array
        The result of regridding the data.

    """
    data = np.moveaxis(data, [grid_x_dim, grid_y_dim], [-1, -2])
    result = regridder.regrid(data, mdtol=mdtol)

    result = np.moveaxis(result, [-1, -2], [grid_x_dim, grid_y_dim])
    return result


def _regrid_along_dims(regridder, data, dims, num_out_dims, mdtol):
    standard_in_dims = [-1, -2][: len(dims)]
    data = np.moveaxis(data, dims, standard_in_dims)
    result = regridder.regrid(data, mdtol=mdtol)
    num_dims = len(dims)

    standard_out_dims = [-1, -2][:num_out_dims]
    if num_dims == 2 and num_out_dims == 1:
        dims = [min(dims)]
    if num_dims == 1 and num_out_dims == 2:
        dims = [dims[0] + 1, dims[0]]
    result = np.moveaxis(result, standard_out_dims, dims)
    return result


def _map_complete_blocks(src, func, dims, out_sizes):
    """
    Apply a function to complete blocks.

    Based on :func:`iris._lazy_data.map_complete_blocks`.
    By "complete blocks" we mean that certain dimensions are enforced to be
    spanned by single chunks.
    Unlike the iris version of this function, this function also handles
    cases where the input and output have a different number of dimensions.
    The particular cases this function is designed for involves collapsing
    a 2D grid to a 1D mesh and expanding a 1D mesh to a 2D grid. Cases
    involving the mapping between the same number of dimensions should still
    behave the same as before.

    Parameters
    ----------
    src : cube
        Source :class:`~iris.cube.Cube` that function is applied to.
    func : function
        Function to apply.
    dims : tuple of int
        Dimensions that cannot be chunked.
    out_sizes : tuple of int
        Output size of dimensions that cannot be chunked.

    Returns
    -------
    array
        Either a :class:`dask.array.array`, or :class:`numpy.ndarray`
        depending on the laziness of the data in src.

    """
    if not src.has_lazy_data():
        return func(src.data)

    data = src.lazy_data()

    # Ensure dims are not chunked
    in_chunks = list(data.chunks)
    for dim in dims:
        in_chunks[dim] = src.shape[dim]
    data = data.rechunk(in_chunks)

    # Determine output chunks
    num_dims = len(dims)
    num_out = len(out_sizes)
    out_chunks = list(data.chunks)
    sorted_dims = sorted(dims)
    if num_out == 1:
        out_chunks[sorted_dims[0]] = out_sizes[0]
    else:
        for dim, size in zip(dims, out_sizes):
            out_chunks[dim] = size

    dropped_dims = []
    new_axis = None
    if num_out > num_dims:
        # While this code should be robust for cases where num_out > num_dims > 1,
        # there is some ambiguity as to what their behaviour ought to be.
        # Since these cases are out of our own scope, we explicitly ignore them
        # for the time being.
        assert num_dims == 1
        # While this code should be robust for cases where num_out > 2,
        # we expect to handle at most 2D grids.
        # Since these cases are out of our own scope, we explicitly ignore them
        # for the time being.
        assert num_out == 2
        slice_index = sorted_dims[-1]
        # Insert the remaining contents of out_sizes in the position immediately
        # after the last dimension.
        out_chunks[slice_index:slice_index] = out_sizes[num_dims:]
        new_axis = range(slice_index, slice_index + num_out - num_dims)
    elif num_dims > num_out:
        # While this code should be robust for cases where num_dims > num_out > 1,
        # there is some ambiguity as to what their behaviour ought to be.
        # Since these cases are out of our own scope, we explicitly ignore them
        # for the time being.
        assert num_out == 1
        # While this code should be robust for cases where num_dims > 2,
        # we expect to handle at most 2D grids.
        # Since these cases are out of our own scope, we explicitly ignore them
        # for the time being.
        assert num_dims == 2
        dropped_dims = sorted_dims[num_out:]
        # Remove the remaining dimensions from the expected output shape.
        for dim in dropped_dims[::-1]:
            out_chunks.pop(dim)
    else:
        pass

    return data.map_blocks(
        func,
        chunks=out_chunks,
        drop_axis=dropped_dims,
        new_axis=new_axis,
        dtype=src.dtype,
    )


def _create_cube(data, src_cube, src_dims, tgt_coords, num_tgt_dims):
    r"""
    Return a new cube for the result of regridding.

    Returned cube represents the result of regridding the source cube
    onto the new grid/mesh.
    All the metadata and coordinates of the result cube are copied from
    the source cube, with two exceptions:
        - Grid/mesh coordinates are copied from the target cube.
        - Auxiliary coordinates which span the grid dimensions are
          ignored.

    Parameters
    ----------
    data : array
        The regridded data as an N-dimensional NumPy array.
    src_cube : cube
        The source Cube.
    src_dims : tuple of int
        The dimensions of the X and Y coordinate within the source Cube.
    tgt_coords : tuple of :class:`iris.coords.Coord`\\ 's
        Either two 1D :class:`iris.coords.DimCoord`\\ 's, two 1D
        :class:`iris.experimental.ugrid.DimCoord`\\ 's or two 2D
        :class:`iris.coords.AuxCoord`\\ 's representing the new grid's
        X and Y coordinates.
    num_tgt_dims : int
        The number of dimensions that the target grid/mesh spans.

    Returns
    -------
    cube
        A new iris.cube.Cube instance.

    """
    new_cube = iris.cube.Cube(data)

    if len(src_dims) == 2:
        grid_x_dim, grid_y_dim = src_dims
    elif len(src_dims) == 1:
        grid_y_dim = src_dims[0]
        grid_x_dim = grid_y_dim + 1
    else:
        raise ValueError(
            f"Source grid must be described by 1 or 2 dimensions, got {len(src_dims)}"
        )
    if num_tgt_dims == 1:
        grid_y_dim = grid_x_dim = min(src_dims)
    for tgt_coord, dim in zip(tgt_coords, (grid_x_dim, grid_y_dim)):
        if len(tgt_coord.shape) == 1:
            if isinstance(tgt_coord, iris.coords.DimCoord):
                new_cube.add_dim_coord(tgt_coord, dim)
            else:
                new_cube.add_aux_coord(tgt_coord, dim)
        else:
            new_cube.add_aux_coord(tgt_coord, (grid_y_dim, grid_x_dim))

    new_cube.metadata = copy.deepcopy(src_cube.metadata)

    def copy_coords(src_coords, add_method):
        for coord in src_coords:
            dims = src_cube.coord_dims(coord)
            if set(src_dims).intersection(set(dims)):
                continue
            offset = num_tgt_dims - len(src_dims)
            dims = [dim if dim < max(src_dims) else dim + offset for dim in dims]
            result_coord = coord.copy()
            # Add result_coord to the owner of add_method.
            add_method(result_coord, dims)

    copy_coords(src_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(src_cube.aux_coords, new_cube.add_aux_coord)

    return new_cube


RegridInfo = namedtuple(
    "RegridInfo", ["x_dim", "y_dim", "x_coord", "y_coord", "regridder"]
)
_RegridInfo = namedtuple("RegridInfo", ["dims", "target", "regridder"])


def _regrid_rectilinear_to_rectilinear__prepare(
    src_grid_cube,
    tgt_grid_cube,
    method,
    precomputed_weights=None,
    srcres=None,
    tgtres=None,
    src_mask=None,
    tgt_mask=None,
):
    tgt_x = _get_coord(tgt_grid_cube, "x")
    tgt_y = _get_coord(tgt_grid_cube, "y")
    src_x = _get_coord(src_grid_cube, "x")
    src_y = _get_coord(src_grid_cube, "y")

    if len(src_x.shape) == 1:
        grid_x_dim = src_grid_cube.coord_dims(src_x)[0]
        grid_y_dim = src_grid_cube.coord_dims(src_y)[0]
    else:
        grid_y_dim, grid_x_dim = src_grid_cube.coord_dims(src_x)

    if method == "conservative":
        center = False
    elif method == "bilinear":
        center = True
    else:
        raise ValueError(
            f"method must be either 'bilinear' or 'conservative', got '{method}'."
        )
    srcinfo = _cube_to_GridInfo(src_grid_cube, center=center, resolution=srcres, mask=src_mask)
    tgtinfo = _cube_to_GridInfo(tgt_grid_cube, center=center, resolution=tgtres, mask=tgt_mask)

    regridder = Regridder(
        srcinfo, tgtinfo, method=method, precomputed_weights=precomputed_weights
    )

    regrid_info = _RegridInfo(
        dims=[grid_x_dim, grid_y_dim],
        target=[tgt_x, tgt_y],
        regridder=regridder,
    )

    return regrid_info


def _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    grid_x_dim, grid_y_dim = regrid_info.dims
    grid_x, grid_y = regrid_info.target
    regridder = regrid_info.regridder

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_dims,
        regridder,
        dims=[grid_x_dim, grid_y_dim],
        num_out_dims=2,
        mdtol=mdtol,
    )

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    if len(grid_x.shape) == 1:
        chunk_shape = (len(grid_y.points), len(grid_x.points))
    else:
        # Due to structural reasons, the order here must be reversed.
        chunk_shape = grid_x.shape[::-1]
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (grid_y_dim, grid_x_dim),
        chunk_shape,
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        (grid_x_dim, grid_y_dim),
        (grid_x, grid_y),
        2,
    )
    return new_cube


def _regrid_unstructured_to_rectilinear__prepare(
    src_mesh_cube,
    target_grid_cube,
    method,
    precomputed_weights=None,
    resolution=None,
):
    """
    First (setup) part of 'regrid_unstructured_to_rectilinear'.

    Check inputs and calculate the sparse regrid matrix and related info.
    The 'regrid info' returned can be re-used over many 2d slices.

    """
    grid_x = _get_coord(target_grid_cube, "x")
    grid_y = _get_coord(target_grid_cube, "y")
    mesh = src_mesh_cube.mesh
    location = src_mesh_cube.location
    if mesh is None:
        raise ValueError("The given cube is not defined on a mesh.")
    if method == "conservative":
        if location != "face":
            raise ValueError(
                f"Conservative regridding requires a source cube located on "
                f"the face of a cube, target cube had the {location} location."
            )
        center = False
    elif method == "bilinear":
        if location not in ["face", "node"]:
            raise ValueError(
                f"Bilinear regridding requires a source cube with a node "
                f"or face location, target cube had the {location} location."
            )
        if location == "face" and None in mesh.face_coords:
            raise ValueError(
                "Bilinear regridding requires a source cube on a face location to have "
                "a face center."
            )
        center = True
    else:
        raise ValueError(
            f"method must be either 'bilinear' or 'conservative', got '{method}'."
        )
    # From src_mesh_cube, fetch the mesh, and the dimension on the cube which that
    # mesh belongs to.
    mesh_dim = src_mesh_cube.mesh_dim()

    meshinfo = _mesh_to_MeshInfo(mesh, location)
    gridinfo = _cube_to_GridInfo(target_grid_cube, center=center, resolution=resolution)

    regridder = Regridder(
        meshinfo, gridinfo, method=method, precomputed_weights=precomputed_weights
    )

    regrid_info = _RegridInfo(
        dims=[mesh_dim],
        target=[grid_x, grid_y],
        regridder=regridder,
    )

    return regrid_info


def _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    """
    Second (regrid) part of 'regrid_unstructured_to_rectilinear'.

    Perform the prepared regrid calculation on a single cube.

    """
    (mesh_dim,) = regrid_info.dims
    grid_x, grid_y = regrid_info.target
    regridder = regrid_info.regridder

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_dims,
        regridder,
        dims=[mesh_dim],
        num_out_dims=2,
        mdtol=mdtol,
    )

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    if len(grid_x.shape) == 1:
        chunk_shape = (len(grid_x.points), len(grid_y.points))
    else:
        chunk_shape = grid_x.shape
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (mesh_dim,),
        chunk_shape,
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        (mesh_dim,),
        (grid_x, grid_y),
        2,
    )

    # TODO: apply tweaks to created cube (slice out length 1 dimensions)

    return new_cube


def _regrid_unstructured_to_unstructured__prepare(
    src_grid_cube,
    target_mesh_cube,
    method,
    precomputed_weights=None,
    resolution=None,
):
    """
    First (setup) part of 'regrid_rectilinear_to_unstructured'.

    Check inputs and calculate the sparse regrid matrix and related info.
    The 'regrid info' returned can be re-used over many 2d slices.

    """
    grid_x = _get_coord(src_grid_cube, "x")
    grid_y = _get_coord(src_grid_cube, "y")
    mesh = target_mesh_cube.mesh
    location = target_mesh_cube.location
    if mesh is None:
        raise ValueError("The given cube is not defined on a mesh.")
    if method == "conservative":
        if location != "face":
            raise ValueError(
                f"Conservative regridding requires a target cube located on "
                f"the face of a cube, target cube had the {location} location."
            )
        center = False
    elif method == "bilinear":
        if location not in ["face", "node"]:
            raise ValueError(
                f"Bilinear regridding requires a target cube with a node "
                f"or face location, target cube had the {location} location."
            )
        if location == "face" and None in mesh.face_coords:
            raise ValueError(
                "Bilinear regridding requires a target cube on a face location to have "
                "a face center."
            )
        center = True
    else:
        raise ValueError(
            f"method must be either 'bilinear' or 'conservative', got '{method}'."
        )
    assert mesh is not None
    if grid_x.ndim == 1:
        (grid_x_dim,) = src_grid_cube.coord_dims(grid_x)
        (grid_y_dim,) = src_grid_cube.coord_dims(grid_y)
    else:
        grid_y_dim, grid_x_dim = src_grid_cube.coord_dims(grid_x)

    meshinfo = _mesh_to_MeshInfo(mesh, location)
    gridinfo = _cube_to_GridInfo(src_grid_cube, center=center, resolution=resolution)

    regridder = Regridder(
        gridinfo, meshinfo, method=method, precomputed_weights=precomputed_weights
    )

    regrid_info = _RegridInfo(
        dims=[grid_x_dim, grid_y_dim],
        target=[mesh, location],
        regridder=regridder,
    )

    return regrid_info


def _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol):
    """
    Second (regrid) part of 'regrid_rectilinear_to_unstructured'.

    Perform the prepared regrid calculation on a single cube.

    """
    grid_x_dim, grid_y_dim = regrid_info.dims
    mesh, location = regrid_info.target
    regridder = regrid_info.regridder

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_dims,
        regridder,
        dims=[grid_x_dim, grid_y_dim],
        num_out_dims=1,
        mdtol=mdtol,
    )
    if location == "face":
        face_node = mesh.face_node_connectivity
        # In face_node_connectivity: `location`= face, `connected` = node, so
        # you want to get the length of the `location` dimension.
        chunk_shape = (face_node.shape[face_node.location_axis],)
    elif location == "node":
        chunk_shape = mesh.node_coords[0].shape

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (grid_y_dim, grid_x_dim),
        chunk_shape,
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        (grid_x_dim, grid_y_dim),
        mesh.to_MeshCoords(location),
        1,
    )
    return new_cube


def _regrid_rectilinear_to_unstructured__prepare(
    src_grid_cube,
    target_mesh_cube,
    method,
    precomputed_weights=None,
    resolution=None,
):
    raise NotImplementedError


def _regrid_unstructured_to_unstructured__perform(src_cube, regrid_info, mdtol):
    raise NotImplementedError


def regrid_rectilinear_to_rectilinear(
    src_cube, grid_cube, mdtol=0, method="conservative", srcres=None, tgtres=None
):
    r"""
    Regrid rectilinear :class:`~iris.cube.Cube` onto another rectilinear grid.

    Return a new :class:`~iris.cube.Cube` with :attr:`~iris.cube.Cube.data`
    values calculated using the area weighted
    mean of :attr:`~iris.cube.Cube.data` values from ``src_cube`` regridded onto the
    horizontal grid of ``grid_cube``.

    Parameters
    ----------
    src_cube : :class:`iris.cube.Cube`
        A rectilinear instance of :class:`~iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    grid_cube : :class:`iris.cube.Cube`
        A rectilinear instance of :class:`~iris.cube.Cube` that supplies the desired
        horizontal grid definition.
    mdtol : float, default=0
        Tolerance of missing data. The value returned in each element of the
        returned :class:`~iris.cube.Cube`\\ 's :attr:`~iris.cube.Cube.data`
        array will be masked if the fraction of masked
        data in the overlapping cells of ``src_cube`` exceeds ``mdtol``. This
        fraction is calculated based on the area of masked cells within each
        target cell. ``mdtol=0`` means no missing data is tolerated while ``mdtol=1``
        will mean the resulting element will be masked if and only if all the
        overlapping cells of ``src_cube`` are masked.
    method : str, default="conservative"
        Either "conservative" or "bilinear". Corresponds to the :mod:`ESMF` methods
        :attr:`~ESMF.api.constants.RegridMethod.CONSERVE` or
        :attr:`~ESMF.api.constants.RegridMethod.BILINEAR` used to calculate weights.
    srcres : int, optional
        If present, represents the amount of latitude slices per source cell
        given to ESMF for calculation.
    tgtres : int, optional
        If present, represents the amount of latitude slices per target cell
        given to ESMF for calculation.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    regrid_info = _regrid_rectilinear_to_rectilinear__prepare(
        src_cube, grid_cube, method=method, srcres=srcres, tgtres=tgtres
    )
    result = _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class ESMFAreaWeighted:
    """
    A scheme which can be recognised by :meth:`iris.cube.Cube.regrid`.

    This class describes an area-weighted regridding scheme for regridding
    between horizontal grids with separated ``X`` and ``Y`` coordinates. It uses
    :mod:`esmpy` to be able to handle grids in different coordinate systems.
    """

    def __init__(self, mdtol=0, use_src_mask=False, use_tgt_mask=False):
        """
        Area-weighted scheme for regridding between rectilinear grids.

        Parameters
        ----------
        mdtol : float, default=0
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of missing data
            exceeds ``mdtol``. This fraction is calculated based on the area of
            masked cells within each target cell. ``mdtol=0`` means no masked
            data is tolerated while ``mdtol=1`` will mean the resulting element
            will be masked if and only if all the overlapping elements of the
            source grid are masked.
        use_src_mask : bool, default=False
            If True, derive a mask from source cube which will tell :mod:`ESMF`
            which points to ignore.
        use_tgt_mask : bool, default=False
            If True, derive a mask from target cube which will tell :mod:`ESMF`
            which points to ignore.

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol
        self.use_src_mask = use_src_mask
        self.use_tgt_mask = use_tgt_mask

    def __repr__(self):
        """Return a representation of the class."""
        return "ESMFAreaWeighted(mdtol={})".format(self.mdtol)

    def regridder(self, src_grid, tgt_grid, use_src_mask=None, use_tgt_mask=None):
        """
        Create regridder to perform regridding from ``src_grid`` to ``tgt_grid``.

        Parameters
        ----------
        src_grid : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` defining the source grid.
        tgt_grid : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` defining the target grid.
        use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, optional
            Array describing which elements :mod:`ESMF` will ignore on the src_grid.
            If True, the mask will be derived from src_grid.
        use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, optional
            Array describing which elements :mod:`ESMF` will ignore on the tgt_grid.
            If True, the mask will be derived from tgt_grid.

        Returns
        -------
        :class:`ESMFAreaWeightedRegridder`
            A callable instance of a regridder with the interface: ``regridder(cube)``
                ... where ``cube`` is a :class:`~iris.cube.Cube` with the same
                grid as ``src_grid`` that is to be regridded to the grid of
                ``tgt_grid``.
        """
        if use_src_mask is None:
            use_src_mask = self.use_src_mask
        if use_tgt_mask is None:
            use_tgt_mask = self.use_tgt_mask
        return ESMFAreaWeightedRegridder(
            src_grid,
            tgt_grid,
            mdtol=self.mdtol,
            use_src_mask=use_src_mask,
            use_tgt_mask=use_tgt_mask,
        )


class _ESMFRegridder:
    r"""Generic regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src,
        tgt,
        method,
        mdtol=None,
        **kwargs,
    ):
        """
        Create regridder for conversions between ``src_grid`` and ``tgt_grid``.

        Parameters
        ----------
        src : :class:`iris.cube.Cube`
            The rectilinear :class:`~iris.cube.Cube` providing the source grid.
        tgt : :class:`iris.cube.Cube`
            The rectilinear :class:`~iris.cube.Cube` providing the target grid.
        mdtol : float, default=0
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
            ``mdtol=1`` will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.

        """
        if mdtol is None:
            if method == "conservative":
                mdtol = 1
            elif method == "bilinear":
                mdtol = 0
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol
        self.method = method

        if src.mesh is None:
            if tgt.mesh is None:
                regrid_info = _regrid_rectilinear_to_rectilinear__prepare(
                    src, tgt, method, **kwargs
                )
            else:
                regrid_info = _regrid_rectilinear_to_unstructured__prepare(
                    src, tgt, method, **kwargs
                )
        else:
            if tgt.mesh is None:
                regrid_info = _regrid_unstructured_to_rectilinear__prepare(
                    src, tgt, method, **kwargs
                )
            else:
                regrid_info = _regrid_unstructured_to_unstructured__prepare(
                    src, tgt, method, **kwargs
                )

        # Store regrid info.
        self.target = regrid_info.target
        self.regridder = regrid_info.regridder

        # Record the source grid.
        if src.mesh is not None:
            self.src = (src.mesh, src.location)
        else:
            self.src = (_get_coord(src, "x"), _get_coord(src, "y"))

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of this regridder instance.

        The given :class:`~iris.cube.Cube` must be defined with the same grid as the source
        :class:`~iris.cube.Cube` used to create this :class:`ESMFAreaWeightedRegridder` instance.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` instance to be regridded.

        Returns
        -------
        :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` defined with the horizontal dimensions of the target
            and the other dimensions from this :class:`~iris.cube.Cube`. The data values of
            this :class:`~iris.cube.Cube` will be converted to values on the new grid using
            area-weighted regridding via :mod:`ESMF` generated weights.

        """
        if cube.mesh is not None:
            src_mesh = cube.mesh
            location = cube.location
            if self.src != (src_mesh, location):
                raise ValueError(
                    "The given cube is not defined on the same "
                    "source mesh as this regridder."
                )
            dims = [cube.mesh_dim()]

        else:
            src_x, src_y = (_get_coord(cube, "x"), _get_coord(cube, "y"))

            # Check the source grid matches that used in initialisation
            if self.src != (src_x, src_y):
                raise ValueError(
                    "The given cube is not defined on the same "
                    "source grid as this regridder."
                )

            if len(src_x.shape) == 1:
                dims = [cube.coord_dims(src_x)[0], cube.coord_dims(src_y)[0]]
            else:
                dims = cube.coord_dims(src_x)

        regrid_info = _RegridInfo(
            dims=dims,
            target=self.target,
            regridder=self.regridder,
        )

        if cube.mesh is None:
            if type(self.target[1]) is not str:
                result = _regrid_rectilinear_to_rectilinear__perform(
                    cube, regrid_info, self.mdtol
                )
            else:
                result = _regrid_rectilinear_to_unstructured__perform(
                    cube, regrid_info, self.mdtol
                )
        else:
            if type(self.target[1]) is not str:
                result = _regrid_unstructured_to_rectilinear__perform(
                    cube, regrid_info, self.mdtol
                )
            else:
                result = _regrid_unstructured_to_unstructured__perform(
                    cube, regrid_info, self.mdtol
                )

        return result


class ESMFAreaWeightedRegridder(_ESMFRegridder):
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src,
        tgt,
        mdtol=0,
        precomputed_weights=None,
        srcres=None,
        tgtres=None,
        resolution=None,
    ):
        kwargs = dict()
        if srcres is not None:
            kwargs["srcres"] = srcres
        if tgtres is not None:
            kwargs["tgtres"] = tgtres
        if resolution is not None:
            kwargs["srcres"] = resolution
        super().__init__(
            src,
            tgt,
            "conservative",
            mdtol=mdtol,
            precomputed_weights=precomputed_weights,
            **kwargs,
        )


class ESMFBilinearRegridder(_ESMFRegridder):
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src,
        tgt,
        mdtol=0,
        precomputed_weights=None,
    ):
        super().__init__(
            src,
            tgt,
            "bilinear",
            mdtol=mdtol,
            precomputed_weights=precomputed_weights,
        )


class _ESMFAreaWeightedRegridder:
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src_grid,
        tgt_grid,
        mdtol=0,
        precomputed_weights=None,
        srcres=None,
        tgtres=None,
        use_src_mask=False,
        use_tgt_mask=False,
    ):
        """
        Create regridder for conversions between ``src_grid`` and ``tgt_grid``.

        Parameters
        ----------
        src_grid : :class:`iris.cube.Cube`
            The rectilinear :class:`~iris.cube.Cube` providing the source grid.
        tgt_grid : :class:`iris.cube.Cube`
            The rectilinear :class:`~iris.cube.Cube` providing the target grid.
        mdtol : float, default=0
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
            ``mdtol=1`` will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
        use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the source to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``src_grid``. If False, no mask will be taken and all points will
            be used in weights calculation.
        use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the source to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``tgt_grid``. If False, no mask will be taken and all points will
            be used in weights calculation.

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        self.src_mask = _get_mask(src_grid, use_src_mask)
        self.tgt_mask = _get_mask(tgt_grid, use_tgt_mask)

        regrid_info = _regrid_rectilinear_to_rectilinear__prepare(
            src_grid, tgt_grid, "conservative", src_mask=self.src_mask, tgt_mask=self.tgt_mask
        )

        # Store regrid info.
        self.grid_x = regrid_info.x_coord
        self.grid_y = regrid_info.y_coord
        self.regridder = regrid_info.regridder

        # Record the source grid.
        self.src_grid = (_get_coord(src_grid, "x"), _get_coord(src_grid, "y"))

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of this regridder instance.

        The given :class:`~iris.cube.Cube` must be defined with the same grid as the source
        :class:`~iris.cube.Cube` used to create this :class:`ESMFAreaWeightedRegridder` instance.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` instance to be regridded.

        Returns
        -------
        :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` defined with the horizontal dimensions of the target
            and the other dimensions from this :class:`~iris.cube.Cube`. The data values of
            this :class:`~iris.cube.Cube` will be converted to values on the new grid using
            area-weighted regridding via :mod:`esmpy` generated weights.

        """
        src_x, src_y = (_get_coord(cube, "x"), _get_coord(cube, "y"))

        # Check the source grid matches that used in initialisation
        if self.src_grid != (src_x, src_y):
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )

        if len(src_x.shape) == 1:
            grid_x_dim = cube.coord_dims(src_x)[0]
            grid_y_dim = cube.coord_dims(src_y)[0]
        else:
            grid_y_dim, grid_x_dim = cube.coord_dims(src_x)

        regrid_info = RegridInfo(
            x_dim=grid_x_dim,
            y_dim=grid_y_dim,
            x_coord=self.grid_x,
            y_coord=self.grid_y,
            regridder=self.regridder,
        )

        return _regrid_rectilinear_to_rectilinear__perform(
            cube, regrid_info, self.mdtol
        )
