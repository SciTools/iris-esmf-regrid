"""Provides an iris interface for unstructured regridding."""

from esmf_regrid.schemes import (
    _ESMFRegridder,
    _get_mask,
    _regrid_rectilinear_to_unstructured__perform,
    _regrid_rectilinear_to_unstructured__prepare,
    _regrid_unstructured_to_rectilinear__perform,
    _regrid_unstructured_to_rectilinear__prepare,
)


def regrid_unstructured_to_rectilinear(
    src_cube,
    grid_cube,
    mdtol=0,
    method="conservative",
    resolution=None,
    use_src_mask=False,
    use_tgt_mask=False,
):
    r"""
    Regrid unstructured :class:`~iris.cube.Cube` onto rectilinear grid.

    Return a new :class:`~iris.cube.Cube` with :attr:`~iris.cube.Cube.data`
    values calculated using weights generated by :mod:`esmpy` to give the weighted
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
        Either "conservative" or "bilinear". Corresponds to the :mod:`esmpy` methods
        :attr:`~esmpy.api.constants.RegridMethod.CONSERVE` or
        :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` used to calculate weights.
    resolution : int, optional
        If present, represents the amount of latitude slices per cell
        given to ESMF for calculation.
    use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
        Either an array representing the cells in the source to ignore, or else
        a boolean value. If True, this array is taken from the mask on the data
        in ``src_mesh_cube``. If False, no mask will be taken and all points will
        be used in weights calculation.
    use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
        Either an array representing the cells in the target to ignore, or else
        a boolean value. If True, this array is taken from the mask on the data
        in ``target_grid_cube``. If False, no mask will be taken and all points
        will be used in weights calculation.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    src_mask = _get_mask(src_cube, use_src_mask)
    tgt_mask = _get_mask(grid_cube, use_tgt_mask)

    regrid_info = _regrid_unstructured_to_rectilinear__prepare(
        src_cube,
        grid_cube,
        method=method,
        resolution=resolution,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )
    result = _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class MeshToGridESMFRegridder(_ESMFRegridder):
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src,
        tgt,
        mdtol=None,
        method="conservative",
        precomputed_weights=None,
        resolution=None,
        use_src_mask=False,
        use_tgt_mask=False,
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
            Either "conservative" or "bilinear". Corresponds to the :mod:`esmpy` methods
            :attr:`~esmpy.api.constants.RegridMethod.CONSERVE` or
            :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` used to calculate weights.
        precomputed_weights : :class:`scipy.sparse.spmatrix`, optional
            If ``None``, :mod:`esmpy` will be used to
            calculate regridding weights. Otherwise, :mod:`esmpy` will be bypassed
            and ``precomputed_weights`` will be used as the regridding weights.
        resolution : int, optional
            If present, represents the amount of latitude slices per cell
            given to ESMF for calculation. If resolution is set, target_grid_cube
            must have strictly increasing bounds (bounds may be transposed plus or
            minus 360 degrees to make the bounds strictly increasing).
        use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the source to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``src_mesh_cube``. If False, no mask will be taken and all points will
            be used in weights calculation.
        use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the target to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``target_grid_cube``. If False, no mask will be taken and all points
            will be used in weights calculation.


        """
        assert src.mesh is not None
        super().__init__(
            src,
            tgt,
            method,
            mdtol=mdtol,
            precomputed_weights=precomputed_weights,
            resolution=resolution,
            use_src_mask=use_src_mask,
            use_tgt_mask=use_tgt_mask,
        )
        self.resolution = resolution
        self.mesh, self.location = self._src
        self.grid_x, self.grid_y = self._tgt


def regrid_rectilinear_to_unstructured(
    src_cube,
    mesh_cube,
    mdtol=0,
    method="conservative",
    resolution=None,
    use_src_mask=False,
    use_tgt_mask=False,
):
    r"""
    Regrid rectilinear :class:`~iris.cube.Cube` onto unstructured mesh.

    Return a new :class:`~iris.cube.Cube` with :attr:`~iris.cube.Cube.data`
    values calculated using weights generated by :mod:`esmpy` to give the weighted
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
        Either "conservative" or "bilinear". Corresponds to the :mod:`esmpy` methods
        :attr:`~esmpy.api.constants.RegridMethod.CONSERVE` or
        :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` used to calculate weights.
    resolution : int, optional
        If present, represents the amount of latitude slices per cell
        given to ESMF for calculation.
    use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
        Either an array representing the cells in the source to ignore, or else
        a boolean value. If True, this array is taken from the mask on the data
        in ``src_mesh_cube``. If False, no mask will be taken and all points will
        be used in weights calculation.
    use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
        Either an array representing the cells in the target to ignore, or else
        a boolean value. If True, this array is taken from the mask on the data
        in ``target_grid_cube``. If False, no mask will be taken and all points
        will be used in weights calculation.

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    src_mask = _get_mask(src_cube, use_src_mask)
    tgt_mask = _get_mask(mesh_cube, use_tgt_mask)

    regrid_info = _regrid_rectilinear_to_unstructured__prepare(
        src_cube,
        mesh_cube,
        method=method,
        resolution=resolution,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )
    result = _regrid_rectilinear_to_unstructured__perform(src_cube, regrid_info, mdtol)
    return result


class GridToMeshESMFRegridder(_ESMFRegridder):
    r"""Regridder class for rectilinear to unstructured :class:`~iris.cube.Cube`\\ s."""

    def __init__(
        self,
        src,
        tgt,
        mdtol=None,
        method="conservative",
        precomputed_weights=None,
        resolution=None,
        use_src_mask=False,
        use_tgt_mask=False,
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
            Either "conservative" or "bilinear". Corresponds to the :mod:`esmpy` methods
            :attr:`~esmpy.api.constants.RegridMethod.CONSERVE` or
            :attr:`~esmpy.api.constants.RegridMethod.BILINEAR` used to calculate weights.
        precomputed_weights : :class:`scipy.sparse.spmatrix`, optional
            If ``None``, :mod:`esmpy` will be used to
            calculate regridding weights. Otherwise, :mod:`esmpy` will be bypassed
            and ``precomputed_weights`` will be used as the regridding weights.
        resolution : int, optional
            If present, represents the amount of latitude slices per cell
            given to ESMF for calculation. If resolution is set, src_grid_cube
            must have strictly increasing bounds (bounds may be transposed plus or
            minus 360 degrees to make the bounds strictly increasing).
        use_src_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the source to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``src_grid_cube``. If False, no mask will be taken and all points will
            be used in weights calculation.
        use_tgt_mask : :obj:`~numpy.typing.ArrayLike` or bool, default=False
            Either an array representing the cells in the target to ignore, or else
            a boolean value. If True, this array is taken from the mask on the data
            in ``target_mesh_cube``. If False, no mask will be taken and all points
            will be used in weights calculation.

        """
        assert tgt.mesh is not None
        super().__init__(
            src,
            tgt,
            method,
            mdtol=mdtol,
            precomputed_weights=precomputed_weights,
            resolution=resolution,
            use_src_mask=use_src_mask,
            use_tgt_mask=use_tgt_mask,
        )
        self.resolution = resolution
        self.mesh, self.location = self._tgt
        self.grid_x, self.grid_y = self._src
