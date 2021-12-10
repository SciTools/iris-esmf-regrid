"""Provides an iris interface for unstructured regridding."""

import copy
import functools

import iris
from iris.analysis._interpolation import get_xy_dim_coords
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo, Regridder
from esmf_regrid.experimental.unstructured_regrid import MeshInfo


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
    sorted_dims = sorted(dims)
    out_chunks = list(data.chunks)
    for dim, size in zip(sorted_dims, out_sizes):
        out_chunks[dim] = size

    num_dims = len(dims)
    num_out = len(out_sizes)
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


# Taken from PR #26
def _bounds_cf_to_simple_1d(cf_bounds):
    assert (cf_bounds[1:, 0] == cf_bounds[:-1, 1]).all()
    simple_bounds = np.empty((cf_bounds.shape[0] + 1,), dtype=np.float64)
    simple_bounds[:-1] = cf_bounds[:, 0]
    simple_bounds[-1] = cf_bounds[-1, 1]
    return simple_bounds


def _mesh_to_MeshInfo(mesh):
    # Returns a MeshInfo object describing the mesh of the cube.
    assert mesh.topology_dimension == 2
    meshinfo = MeshInfo(
        np.stack([coord.points for coord in mesh.node_coords], axis=-1),
        mesh.face_node_connectivity.indices,
        mesh.face_node_connectivity.start_index,
    )
    return meshinfo


def _cube_to_GridInfo(cube):
    # This is a simplified version of an equivalent function/method in PR #26.
    # It is anticipated that this function will be replaced by the one in PR #26.
    #
    # Returns a GridInfo object describing the horizontal grid of the cube.
    # This may be inherited from code written for the rectilinear regridding scheme.
    lon = cube.coord("longitude")
    lat = cube.coord("latitude")
    # Ensure coords come from a proper grid.
    assert isinstance(lon, iris.coords.DimCoord)
    assert isinstance(lat, iris.coords.DimCoord)
    # TODO: accommodate other x/y coords.
    # TODO: perform checks on lat/lon.
    #  Checks may cover units, coord systems (e.g. rotated pole), contiguous bounds.
    return GridInfo(
        lon.points,
        lat.points,
        _bounds_cf_to_simple_1d(lon.bounds),
        _bounds_cf_to_simple_1d(lat.bounds),
        circular=lon.circular,
    )


def _regrid_along_mesh_dim(regridder, data, mesh_dim, mdtol):
    # Before regridding, data is transposed to a standard form.
    # In the future, this may be done within the regridder by specifying args.

    # Move the mesh axis to be the last dimension.
    data = np.moveaxis(data, mesh_dim, -1)

    result = regridder.regrid(data, mdtol=mdtol)

    # Move grid axes back into the original position of the mesh.
    result = np.moveaxis(result, [-2, -1], [mesh_dim, mesh_dim + 1])

    return result


def _create_cube(data, src_cube, mesh_dim, grid_x, grid_y):
    """
    Return a new cube for the result of regridding.

    Returned cube represents the result of regridding the source cube
    onto the new grid.
    All the metadata and coordinates of the result cube are copied from
    the source cube, with two exceptions:
        - Grid dimension coordinates are copied from the grid cube.
        - Auxiliary coordinates which span the grid dimensions are
          ignored.

    Parameters
    ----------
    data : array
        The regridded data as an N-dimensional NumPy array.
    src_cube : cube
        The source Cube.
    mesh_dim : int
        The dimension of the mesh within the source Cube.
    grid_x : DimCoord
        The :class:`iris.coords.DimCoord` for the new grid's X
        coordinate.
    grid_y : DimCoord
        The :class:`iris.coords.DimCoord` for the new grid's Y
        coordinate.

    Returns
    -------
    cube
        A new iris.cube.Cube instance.

    """
    new_cube = iris.cube.Cube(data)

    # TODO: The following code is rigid with respect to which dimensions
    #  the x coord and y coord are assigned to. We should decide if it is
    #  appropriate to copy the dimension ordering from the target cube
    #  instead.
    new_cube.add_dim_coord(grid_x, mesh_dim + 1)
    new_cube.add_dim_coord(grid_y, mesh_dim)

    new_cube.metadata = copy.deepcopy(src_cube.metadata)

    # TODO: Handle derived coordinates. The following code is taken from
    #  iris, the parts dealing with derived coordinates have been
    #  commented out for the time being.
    # coord_mapping = {}

    def copy_coords(src_coords, add_method):
        for coord in src_coords:
            dims = src_cube.coord_dims(coord)
            if hasattr(coord, "mesh") or mesh_dim in dims:
                continue
            # Since the mesh will be replaced by a 2D grid, dims which are
            # beyond the mesh_dim are increased by one.
            dims = [dim if dim < mesh_dim else dim + 1 for dim in dims]
            result_coord = coord.copy()
            # Add result_coord to the owner of add_method.
            add_method(result_coord, dims)
            # coord_mapping[id(coord)] = result_coord

    copy_coords(src_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(src_cube.aux_coords, new_cube.add_aux_coord)

    # for factory in src_cube.aux_factories:
    #     # TODO: Regrid dependant coordinates which span mesh_dim.
    #     try:
    #         result.add_aux_factory(factory.updated(coord_mapping))
    #     except KeyError:
    #         msg = (
    #             "Cannot update aux_factory {!r} because of dropped"
    #             " coordinates.".format(factory.name())
    #         )
    #         warnings.warn(msg)

    return new_cube


def _regrid_unstructured_to_rectilinear__prepare(
    src_mesh_cube, target_grid_cube, precomputed_weights=None
):
    """
    First (setup) part of 'regrid_unstructured_to_rectilinear'.

    Check inputs and calculate the sparse regrid matrix and related info.
    The 'regrid info' returned can be re-used over many 2d slices.

    """
    # TODO: Perform checks on the arguments. (grid coords are contiguous,
    #  spherical and monotonic. Mesh is defined on faces)

    # TODO: Account for differences in units.

    # TODO: Account for differences in coord systems.

    # TODO: Record appropriate dimensions (i.e. which dimension the mesh belongs to)

    grid_x, grid_y = get_xy_dim_coords(target_grid_cube)
    mesh = src_mesh_cube.mesh
    # TODO: Improve the checking of mesh validity. Check the mesh location and
    #  raise appropriate error messages.
    assert mesh is not None
    # From src_mesh_cube, fetch the mesh, and the dimension on the cube which that
    # mesh belongs to.
    mesh_dim = src_mesh_cube.mesh_dim()

    meshinfo = _mesh_to_MeshInfo(mesh)
    gridinfo = _cube_to_GridInfo(target_grid_cube)

    regridder = Regridder(meshinfo, gridinfo, precomputed_weights)

    regrid_info = (mesh_dim, grid_x, grid_y, regridder)

    return regrid_info


def _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    """
    Second (regrid) part of 'regrid_unstructured_to_rectilinear'.

    Perform the prepared regrid calculation on a single cube.

    """
    mesh_dim, grid_x, grid_y, regridder = regrid_info

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_mesh_dim,
        regridder,
        mesh_dim=mesh_dim,
        mdtol=mdtol,
    )

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (mesh_dim,),
        (len(grid_x.points), len(grid_y.points)),
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        mesh_dim,
        grid_x,
        grid_y,
    )

    # TODO: apply tweaks to created cube (slice out length 1 dimensions)

    return new_cube


def regrid_unstructured_to_rectilinear(src_cube, grid_cube, mdtol=0):
    """
    Regrid unstructured cube onto rectilinear grid.

    Return a new cube with data values calculated using the area weighted
    mean of data values from unstructured cube src_cube regridded onto the
    horizontal grid of grid_cube. The dimension on the cube belonging to
    the mesh will replaced by the two dimensions associated with the grid.
    This function requires that the horizontal dimension of src_cube is
    described by a 2D mesh with data located on the faces of that mesh.
    This function requires that the horizontal grid of grid_cube is
    rectilinear (i.e. expressed in terms of two orthogonal 1D coordinates).
    This function also requires that the coordinates describing the
    horizontal grid have bounds.

    Parameters
    ----------
    src_cube : cube
        An unstructured instance of iris.cube.Cube that supplies the data,
        metadata and coordinates.
    grid_cube : cube
        A rectilinear instance of iris.cube.Cube that supplies the desired
        horizontal grid definition.
    mdtol : float, optional
        Tolerance of missing data. The value returned in each element of the
        returned cube's data array will be masked if the fraction of masked
        data in the overlapping cells of the source cube exceeds mdtol. This
        fraction is calculated based on the area of masked cells within each
        target cell. mdtol=0 means no missing data is tolerated while mdtol=1
        will mean the resulting element will be masked if and only if all the
        overlapping cells of the source cube are masked. Defaults to 0.

    Returns
    -------
    cube
        A new iris.cube.Cube instance.

    """
    regrid_info = _regrid_unstructured_to_rectilinear__prepare(src_cube, grid_cube)
    result = _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class MeshToGridESMFRegridder:
    """Regridder class for unstructured to rectilinear cubes."""

    def __init__(
        self, src_mesh_cube, target_grid_cube, mdtol=1, precomputed_weights=None
    ):
        """
        Create regridder for conversions between source mesh and target grid.

        Parameters
        ----------
        src_grid_cube : cube
            The unstructured iris cube providing the source grid.
        target_grid_cube : cube
            The rectilinear iris cube providing the target grid.
        mdtol : float, optional
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 1.

        """
        # TODO: Record information about the identity of the mesh. This would
        #  typically be a copy of the mesh, though given the potential size of
        #  the mesh, it may make sense to either retain a reference to the actual
        #  mesh or else something like a hash of the mesh.

        # Missing data tolerance.
        # Code directly copied from iris.
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        partial_regrid_info = _regrid_unstructured_to_rectilinear__prepare(
            src_mesh_cube, target_grid_cube, precomputed_weights=precomputed_weights
        )

        # Record source mesh.
        self.mesh = src_mesh_cube.mesh

        # Store regrid info.
        _, self.grid_x, self.grid_y, self.regridder = partial_regrid_info

    def __call__(self, cube):
        """
        Regrid this cube onto the target grid of this regridder instance.

        The given cube must be defined with the same mesh as the source
        cube used to create this MeshToGridESMFRegridder instance.

        Parameters
        ----------
        cube : cube
            A iris.cube.Cube instance to be regridded.

        Returns
        -------
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding via ESMF generated weights.

        """
        mesh = cube.mesh
        self_mesh = copy.deepcopy(self.mesh)
        self_mesh.var_name = mesh.var_name
        for self_coord, other_coord in zip(self_mesh.all_coords, mesh.all_coords):
            if self_coord is not None:
                self_coord.var_name = other_coord.var_name
        for self_con, other_con in zip(self_mesh.all_connectivities, mesh.all_connectivities):
            if self_con is not None:
                self_con.var_name = other_con.var_name
        if self_mesh != mesh:
            raise ValueError(
                "The given cube is not defined on the same "
                "source mesh as this regridder."
            )

        mesh_dim = cube.mesh_dim()

        regrid_info = (mesh_dim, self.grid_x, self.grid_y, self.regridder)

        return _regrid_unstructured_to_rectilinear__perform(
            cube, regrid_info, self.mdtol
        )


def _regrid_along_grid_dims(regridder, data, grid_x_dim, grid_y_dim, mdtol):
    # The mesh will be assigned to the first dimension associated with the
    # grid, whether that is associated with the x or y coordinate.
    tgt_mesh_dim = min(grid_x_dim, grid_y_dim)
    data = np.moveaxis(data, [grid_x_dim, grid_y_dim], [-1, -2])
    result = regridder.regrid(data, mdtol=mdtol)

    result = np.moveaxis(result, -1, tgt_mesh_dim)
    return result


def _create_mesh_cube(data, src_cube, grid_x_dim, grid_y_dim, mesh):
    """
    Return a new cube for the result of regridding.

    Returned cube represents the result of regridding the source cube
    onto the new mesh.
    All the metadata and coordinates of the result cube are copied from
    the source cube, with two exceptions:
        - Grid dimension coordinates are copied from the grid cube.
        - Auxiliary coordinates which span the mesh dimension are
          ignored.

    Parameters
    ----------
    data : array
        The regridded data as an N-dimensional NumPy array.
    src_cube : cube
        The source Cube.
    grid_x_dim : int
        The dimension of the x dimension on the source Cube.
    grid_y_dim : int
        The dimension of the y dimension on the source Cube.
    mesh : Mesh
        The :class:`iris.experimental.ugrid.Mesh` for the new
        Cube.

    Returns
    -------
    cube
        A new iris.cube.Cube instance.

    """
    new_cube = iris.cube.Cube(data)

    min_grid_dim = min(grid_x_dim, grid_y_dim)
    max_grid_dim = max(grid_x_dim, grid_y_dim)
    for coord in mesh.to_MeshCoords("face"):
        new_cube.add_aux_coord(coord, min_grid_dim)

    new_cube.metadata = copy.deepcopy(src_cube.metadata)

    def copy_coords(src_coords, add_method):
        for coord in src_coords:
            dims = src_cube.coord_dims(coord)
            if grid_x_dim in dims or grid_y_dim in dims:
                continue
            # Since the 2D grid will be replaced by a 1D mesh, dims which are
            # beyond the max_grid_dim are decreased by one.
            dims = [dim if dim < max_grid_dim else dim - 1 for dim in dims]
            result_coord = coord.copy()
            # Add result_coord to the owner of add_method.
            add_method(result_coord, dims)

    copy_coords(src_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(src_cube.aux_coords, new_cube.add_aux_coord)

    return new_cube


def _regrid_rectilinear_to_unstructured__prepare(
    src_grid_cube, target_mesh_cube, precomputed_weights=None
):
    """
    First (setup) part of 'regrid_rectilinear_to_unstructured'.

    Check inputs and calculate the sparse regrid matrix and related info.
    The 'regrid info' returned can be re-used over many 2d slices.

    """
    grid_x, grid_y = get_xy_dim_coords(src_grid_cube)
    mesh = target_mesh_cube.mesh
    # TODO: Improve the checking of mesh validity. Check the mesh location and
    #  raise appropriate error messages.
    assert mesh is not None
    grid_x_dim = src_grid_cube.coord_dims(grid_x)[0]
    grid_y_dim = src_grid_cube.coord_dims(grid_y)[0]

    meshinfo = _mesh_to_MeshInfo(mesh)
    gridinfo = _cube_to_GridInfo(src_grid_cube)

    regridder = Regridder(gridinfo, meshinfo, precomputed_weights)

    regrid_info = (grid_x_dim, grid_y_dim, grid_x, grid_y, mesh, regridder)

    return regrid_info


def _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol):
    """
    Second (regrid) part of 'regrid_rectilinear_to_unstructured'.

    Perform the prepared regrid calculation on a single cube.

    """
    grid_x_dim, grid_y_dim, grid_x, grid_y, mesh, regridder = regrid_info

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_grid_dims,
        regridder,
        grid_x_dim=grid_x_dim,
        grid_y_dim=grid_y_dim,
        mdtol=mdtol,
    )

    face_node = mesh.face_node_connectivity
    # In face_node_connectivity: `src`= face, `tgt` = node, so you want to
    # get the length of the `src` dimension.
    n_faces = face_node.shape[face_node.src_dim]

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (grid_x_dim, grid_y_dim),
        (n_faces,),
    )

    new_cube = _create_mesh_cube(
        new_data,
        src_cube,
        grid_x_dim,
        grid_y_dim,
        mesh,
    )
    return new_cube


def regrid_rectilinear_to_unstructured(src_cube, mesh_cube, mdtol=0):
    """
    Regrid rectilinear cube onto unstructured mesh.

    Return a new cube with data values calculated using the area weighted
    mean of data values from rectilinear cube src_cube regridded onto the
    horizontal mesh of mesh_cube. The dimensions on the cube associated
    with the grid will replaced by a dimension associated with the mesh.
    That dimension will be the the first of the grid dimensions, whether
    it is associated with the x or y coordinate. Since two dimensions are
    being replaced by one, coordinates associated with dimensions after
    the grid will become associated with dimensions one lower.
    This function requires that the horizontal dimension of mesh_cube is
    described by a 2D mesh with data located on the faces of that mesh.
    This function requires that the horizontal grid of src_cube is
    rectilinear (i.e. expressed in terms of two orthogonal 1D coordinates).
    This function also requires that the coordinates describing the
    horizontal grid have bounds.

    Parameters
    ----------
    src_cube : cube
        A rectilinear instance of iris.cube.Cube that supplies the data,
        metadata and coordinates.
    mesh_cube : cube
        An unstructured instance of iris.cube.Cube that supplies the desired
        horizontal mesh definition.
    mdtol : float, optional
        Tolerance of missing data. The value returned in each element of the
        returned cube's data array will be masked if the fraction of masked
        data in the overlapping cells of the source cube exceeds mdtol. This
        fraction is calculated based on the area of masked cells within each
        target cell. mdtol=0 means no missing data is tolerated while mdtol=1
        will mean the resulting element will be masked if and only if all the
        overlapping cells of the source cube are masked. Defaults to 0.

    Returns
    -------
    cube
        A new iris.cube.Cube instance.

    """
    regrid_info = _regrid_rectilinear_to_unstructured__prepare(src_cube, mesh_cube)
    result = _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol)
    return result


class GridToMeshESMFRegridder:
    """Regridder class for rectilinear to unstructured cubes."""

    def __init__(
        self, src_mesh_cube, target_grid_cube, mdtol=1, precomputed_weights=None
    ):
        """
        Create regridder for conversions between source grid and target mesh.

        Parameters
        ----------
        src_grid_cube : cube
            The unstructured iris cube providing the source grid.
        target_grid_cube : cube
            The rectilinear iris cube providing the target mesh.
        mdtol : float, optional
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 1.

        """
        # Missing data tolerance.
        # Code directly copied from iris.
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        partial_regrid_info = _regrid_rectilinear_to_unstructured__prepare(
            src_mesh_cube, target_grid_cube, precomputed_weights=precomputed_weights
        )

        # Store regrid info.
        _, _, self.grid_x, self.grid_y, self.mesh, self.regridder = partial_regrid_info

    def __call__(self, cube):
        """
        Regrid this cube onto the target mesh of this regridder instance.

        The given cube must be defined with the same grid as the source
        cube used to create this MeshToGridESMFRegridder instance.

        Parameters
        ----------
        cube : cube
            A iris.cube.Cube instance to be regridded.

        Returns
        -------
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            area-weighted regridding via ESMF generated weights.

        """
        grid_x, grid_y = get_xy_dim_coords(cube)
        if (grid_x != self.grid_x) or (grid_y != self.grid_y):
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )

        grid_x_dim = cube.coord_dims(grid_x)[0]
        grid_y_dim = cube.coord_dims(grid_y)[0]

        regrid_info = (
            grid_x_dim,
            grid_y_dim,
            self.grid_x,
            self.grid_y,
            self.mesh,
            self.regridder,
        )

        return _regrid_rectilinear_to_unstructured__perform(
            cube, regrid_info, self.mdtol
        )
