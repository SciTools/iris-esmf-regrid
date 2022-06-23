"""Provides an iris interface for unstructured regridding."""

import copy
import functools

import numpy as np

from esmf_regrid.esmf_regridder import Regridder
from esmf_regrid.experimental.unstructured_regrid import MeshInfo
from esmf_regrid.schemes import _create_cube, _cube_to_GridInfo


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


def _regrid_along_mesh_dim(regridder, data, mesh_dim, mdtol):
    # Before regridding, data is transposed to a standard form.
    # In the future, this may be done within the regridder by specifying args.

    # Move the mesh axis to be the last dimension.
    data = np.moveaxis(data, mesh_dim, -1)

    result = regridder.regrid(data, mdtol=mdtol)

    # Move grid axes back into the original position of the mesh.
    result = np.moveaxis(result, [-2, -1], [mesh_dim, mesh_dim + 1])

    return result


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
    grid_x = target_grid_cube.coord(axis="x")
    grid_y = target_grid_cube.coord(axis="y")
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


def regrid_unstructured_to_rectilinear(
    src_cube,
    grid_cube,
    mdtol=0,
    method="conservative",
    resolution=None,
):
    r"""
    Regrid unstructured :class:`~iris.cube.Cube` onto rectilinear grid.

    Return a new :class:`~iris.cube.Cube` with :attr:`~iris.cube.Cube.data`
    values calculated using weights generated by :mod:`ESMF` to give the weighted
    mean of :attr:`~iris.cube.Cube.data` values from ``src_cube`` regridded onto the
    horizontal grid of ``grid_cube``. The dimension on the :class:`~iris.cube.Cube`
    belonging to the :attr:`~iris.cube.Cube.mesh`
    will replaced by the two dimensions associated with the grid.
    This function requires that the horizontal dimension of ``src_cube`` is
    described by a 2D mesh with data located on the faces of that mesh
    for conservative regridding and located on either faces or nodes for
    bilinear regridding.
    This function allows the horizontal grid of ``grid_cube`` to be either
    rectilinear or curvilinear (i.e. expressed in terms of two orthogonal
    1D coordinates or via a pair of 2D coordinates).
    This function also requires that the :class:`~iris.coords.Coord`\\ s describing the
    horizontal grid have :attr:`~iris.coords.Coord.bounds`.

    Parameters
    ----------
    src_cube : :class:`iris.cube.Cube`
        An unstructured instance of :class:`~iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    grid_cube : :class:`iris.cube.Cube`
        An instance of :class:`~iris.cube.Cube` that supplies the desired
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
    resolution : int, optional
        If present, represents the amount of latitude slices per cell
        given to ESMF for calculation.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    regrid_info = _regrid_unstructured_to_rectilinear__prepare(
        src_cube,
        grid_cube,
        method=method,
        resolution=resolution,
    )
    result = _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class MeshToGridESMFRegridder:
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src_mesh_cube,
        target_grid_cube,
        mdtol=None,
        method="conservative",
        precomputed_weights=None,
        resolution=None,
    ):
        """
        Create regridder for conversions between source mesh and target grid.

        Parameters
        ----------
        src_mesh_cube : :class:`iris.cube.Cube`
            The unstructured :class:`~iris.cube.Cube` providing the source mesh.
        target_grid_cube : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` providing the target grid.
        mdtol : float, optional
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
            ``mdtol=1`` will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked. Defaults to 1
            for conservative regregridding and 0 for bilinear regridding.
        method : str, default="conservative"
            Either "conservative" or "bilinear". Corresponds to the :mod:`ESMF` methods
            :attr:`~ESMF.api.constants.RegridMethod.CONSERVE` or
            :attr:`~ESMF.api.constants.RegridMethod.BILINEAR` used to calculate weights.
        precomputed_weights : :class:`scipy.sparse.spmatrix`, optional
            If ``None``, :mod:`ESMF` will be used to
            calculate regridding weights. Otherwise, :mod:`ESMF` will be bypassed
            and ``precomputed_weights`` will be used as the regridding weights.
        resolution : int, optional
            If present, represents the amount of latitude slices per cell
            given to ESMF for calculation. If resolution is set, target_grid_cube
            must have strictly increasing bounds (bounds may be transposed plus or
            minus 360 degrees to make the bounds strictly increasing).

        """
        # TODO: Record information about the identity of the mesh. This would
        #  typically be a copy of the mesh, though given the potential size of
        #  the mesh, it may make sense to either retain a reference to the actual
        #  mesh or else something like a hash of the mesh.
        if method not in ["conservative", "bilinear"]:
            raise ValueError(
                f"method must be either 'bilinear' or 'conservative', got '{method}'."
            )

        # Missing data tolerance.
        # Code directly copied from iris.
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

        if resolution is not None:
            if not (isinstance(resolution, int) and resolution > 0):
                raise ValueError("resolution must be a positive integer.")
            if method != "conservative":
                raise ValueError(
                    "resolution can only be set for conservative regridding."
                )
        self.resolution = resolution

        partial_regrid_info = _regrid_unstructured_to_rectilinear__prepare(
            src_mesh_cube,
            target_grid_cube,
            method=self.method,
            precomputed_weights=precomputed_weights,
            resolution=resolution,
        )

        # Record source mesh.
        self.mesh = src_mesh_cube.mesh
        self.location = src_mesh_cube.location

        # Store regrid info.
        _, self.grid_x, self.grid_y, self.regridder = partial_regrid_info

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target grid of this regridder instance.

        The given :class:`~iris.cube.Cube` must be defined with the same mesh as the source
        cube used to create this :class:`MeshToGridESMFRegridder` instance.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` instance to be regridded.

        Returns
        -------
        :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` defined with the horizontal dimensions of the target
            and the other dimensions from ``cube``. The
            :attr:`~iris.cube.Cube.data` values of
            ``cube`` will be converted to values on the new grid using
            :mod:`ESMF` generated weights.

        """
        mesh = cube.mesh
        if mesh is None:
            raise ValueError("The given cube is not defined on a mesh.")
        if cube.location != self.location:
            raise ValueError(
                "The given cube is not defined on a the same "
                "mesh location as this regridder."
            )
        # TODO: replace temporary hack when iris issues are sorted.
        # Ignore differences in var_name that might be caused by saving.
        # TODO: uncomment this when iris issue with masked array comparison is sorted.
        # self_mesh = copy.deepcopy(self.mesh)
        # self_mesh.var_name = mesh.var_name
        # for self_coord, other_coord in zip(self_mesh.all_coords, mesh.all_coords):
        #     if self_coord is not None:
        #         self_coord.var_name = other_coord.var_name
        # for self_con, other_con in zip(
        #     self_mesh.all_connectivities, mesh.all_connectivities
        # ):
        #     if self_con is not None:
        #         self_con.var_name = other_con.var_name
        # if self_mesh != mesh:
        #     raise ValueError(
        #         "The given cube is not defined on the same "
        #         "source mesh as this regridder."
        #     )

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


def _regrid_rectilinear_to_unstructured__prepare(
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
    grid_x = src_grid_cube.coord(axis="x")
    grid_y = src_grid_cube.coord(axis="y")
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

    regrid_info = (grid_x_dim, grid_y_dim, grid_x, grid_y, mesh, location, regridder)

    return regrid_info


def _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol):
    """
    Second (regrid) part of 'regrid_rectilinear_to_unstructured'.

    Perform the prepared regrid calculation on a single cube.

    """
    grid_x_dim, grid_y_dim, grid_x, grid_y, mesh, location, regridder = regrid_info

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_grid_dims,
        regridder,
        grid_x_dim=grid_x_dim,
        grid_y_dim=grid_y_dim,
        mdtol=mdtol,
    )

    face_node = mesh.face_node_connectivity
    # In face_node_connectivity: `location`= face, `connected` = node, so
    # you want to get the length of the `location` dimension.
    n_faces = face_node.shape[face_node.location_axis]

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    new_data = _map_complete_blocks(
        src_cube,
        regrid,
        (grid_x_dim, grid_y_dim),
        (n_faces,),
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        (grid_x_dim, grid_y_dim),
        mesh.to_MeshCoords(location),
        1,
    )
    return new_cube


def regrid_rectilinear_to_unstructured(
    src_cube,
    mesh_cube,
    mdtol=0,
    method="conservative",
    resolution=None,
):
    r"""
    Regrid rectilinear :class:`~iris.cube.Cube` onto unstructured mesh.

    Return a new :class:`~iris.cube.Cube` with :attr:`~iris.cube.Cube.data`
    values calculated using weights generated by :mod:`ESMF` to give the weighted
    mean of :attr:`~iris.cube.Cube.data` values from ``src_cube`` regridded onto the
    horizontal mesh of ``mesh_cube``. The dimensions on the :class:`~iris.cube.Cube` associated
    with the grid will replaced by a dimension associated with the
    :attr:`~iris.cube.Cube.mesh`.
    That dimension will be the the first of the grid dimensions, whether
    it is associated with the ``x`` or ``y`` coordinate. Since two dimensions are
    being replaced by one, coordinates associated with dimensions after
    the grid will become associated with dimensions one lower.
    This function requires that the horizontal dimension of ``mesh_cube`` is
    described by a 2D mesh with data located on the faces of that mesh
    for conservative regridding and located on either faces or nodes for
    bilinear regridding.
    This function allows the horizontal grid of ``grid_cube`` to be either
    rectilinear or curvilinear (i.e. expressed in terms of two orthogonal
    1D coordinates or via a pair of 2D coordinates).
    This function also requires that the :class:`~iris.coords.Coord`\\ s describing the
    horizontal grid have :attr:`~iris.coords.Coord.bounds`.

    Parameters
    ----------
    src_cube : :class:`iris.cube.Cube`
        A rectilinear instance of :class:`~iris.cube.Cube` that supplies the data,
        metadata and coordinates.
    mesh_cube : :class:`iris.cube.Cube`
        An unstructured instance of :class:`~iris.cube.Cube` that supplies the desired
        horizontal mesh definition.
    mdtol : float, default=0
        Tolerance of missing data. The value returned in each element of the
        returned :class:`~iris.cube.Cube`\\ 's :attr:`~iris.cube.Cube.data` array
        will be masked if the fraction of masked
        data in the overlapping cells of the source cube exceeds ``mdtol``. This
        fraction is calculated based on the area of masked cells within each
        target cell. ``mdtol=0`` means no missing data is tolerated while ``mdtol=1``
        will mean the resulting element will be masked if and only if all the
        overlapping cells of the ``src_cube`` are masked.
    method : str, default="conservative"
        Either "conservative" or "bilinear". Corresponds to the :mod:`ESMF` methods
        :attr:`~ESMF.api.constants.RegridMethod.CONSERVE` or
        :attr:`~ESMF.api.constants.RegridMethod.BILINEAR` used to calculate weights.
    resolution : int, optional
        If present, represents the amount of latitude slices per cell
        given to ESMF for calculation.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    regrid_info = _regrid_rectilinear_to_unstructured__prepare(
        src_cube,
        mesh_cube,
        method=method,
        resolution=resolution,
    )
    result = _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol)
    return result


class GridToMeshESMFRegridder:
    r"""Regridder class for rectilinear to unstructured :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src_grid_cube,
        target_mesh_cube,
        mdtol=None,
        method="conservative",
        precomputed_weights=None,
        resolution=None,
    ):
        """
        Create regridder for conversions between source grid and target mesh.

        Parameters
        ----------
        src_grid_cube : :class:`iris.cube.Cube`
            The rectilinear :class:`~iris.cube.Cube` cube providing the source grid.
        target_mesh_cube : :class:`iris.cube.Cube`
            The unstructured :class:`~iris.cube.Cube` providing the target mesh.
        mdtol : float, optional
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds ``mdtol``. ``mdtol=0`` means no missing data is tolerated while
            ``mdtol=1`` will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked. Defaults to 1
            for conservative regregridding and 0 for bilinear regridding.
        method : str, default="conservative"
            Either "conservative" or "bilinear". Corresponds to the :mod:`ESMF` methods
            :attr:`~ESMF.api.constants.RegridMethod.CONSERVE` or
            :attr:`~ESMF.api.constants.RegridMethod.BILINEAR` used to calculate weights.
        precomputed_weights : :class:`scipy.sparse.spmatrix`, optional
            If ``None``, :mod:`ESMF` will be used to
            calculate regridding weights. Otherwise, :mod:`ESMF` will be bypassed
            and ``precomputed_weights`` will be used as the regridding weights.
        resolution : int, optional
            If present, represents the amount of latitude slices per cell
            given to ESMF for calculation. If resolution is set, src_grid_cube
            must have strictly increasing bounds (bounds may be transposed plus or
            minus 360 degrees to make the bounds strictly increasing).

        """
        if method not in ["conservative", "bilinear"]:
            raise ValueError(
                f"method must be either 'bilinear' or 'conservative', got '{method}'."
            )

        if resolution is not None:
            if not (isinstance(resolution, int) and resolution > 0):
                raise ValueError("resolution must be a positive integer.")
            if method != "conservative":
                raise ValueError(
                    "resolution can only be set for conservative regridding."
                )
        # Missing data tolerance.
        # Code directly copied from iris.
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
        self.resolution = resolution

        partial_regrid_info = _regrid_rectilinear_to_unstructured__prepare(
            src_grid_cube,
            target_mesh_cube,
            method=self.method,
            precomputed_weights=precomputed_weights,
            resolution=self.resolution,
        )

        # Store regrid info.
        (
            _,
            _,
            self.grid_x,
            self.grid_y,
            self.mesh,
            self.location,
            self.regridder,
        ) = partial_regrid_info

    def __call__(self, cube):
        """
        Regrid this :class:`~iris.cube.Cube` onto the target mesh of this regridder instance.

        The given :class:`~iris.cube.Cube` must be defined with the same grid as the source
        cube used to create this :class:`MeshToGridESMFRegridder` instance.

        Parameters
        ----------
        cube : :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` instance to be regridded.

        Returns
        -------
        :class:`iris.cube.Cube`
            A :class:`~iris.cube.Cube` defined with the horizontal dimensions of the target
            and the other dimensions from ``cube``. The
            :attr:`~iris.cube.Cube.data` values of
            ``cube`` will be converted to values on the new grid using
            area-weighted regridding via :mod:`ESMF` generated weights.

        """
        grid_x = cube.coord(axis="x")
        grid_y = cube.coord(axis="y")
        # Ignore differences in var_name that might be caused by saving.
        self_grid_x = copy.deepcopy(self.grid_x)
        self_grid_x.var_name = grid_x.var_name
        self_grid_y = copy.deepcopy(self.grid_y)
        self_grid_y.var_name = grid_y.var_name
        if (grid_x != self_grid_x) or (grid_y != self_grid_y):
            raise ValueError(
                "The given cube is not defined on the same "
                "source grid as this regridder."
            )

        if len(grid_x.shape) == 1:
            grid_x_dim = cube.coord_dims(grid_x)[0]
            grid_y_dim = cube.coord_dims(grid_y)[0]
        else:
            grid_y_dim, grid_x_dim = cube.coord_dims(grid_x)

        regrid_info = (
            grid_x_dim,
            grid_y_dim,
            self.grid_x,
            self.grid_y,
            self.mesh,
            self.location,
            self.regridder,
        )

        return _regrid_rectilinear_to_unstructured__perform(
            cube, regrid_info, self.mdtol
        )
