"""Provides an iris interface for regridding."""

from collections import namedtuple
import copy
import functools

from cf_units import Unit
from iris._lazy_data import map_complete_blocks
import iris.coords
import iris.cube
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo, RefinedGridInfo, Regridder

__all__ = [
    "ESMFAreaWeighted",
    "ESMFAreaWeightedRegridder",
    "regrid_rectilinear_to_rectilinear",
]


def _cube_to_GridInfo(cube, center=False, resolution=None):
    # This is a simplified version of an equivalent function/method in PR #26.
    # It is anticipated that this function will be replaced by the one in PR #26.
    #
    # Returns a GridInfo object describing the horizontal grid of the cube.
    # This may be inherited from code written for the rectilinear regridding scheme.
    lon, lat = cube.coord(axis="x"), cube.coord(axis="y")
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
        # TODO: perform checks on lat/lon.
    elif londim == 2:
        assert cube.coord_dims(lon) == cube.coord_dims(lat)
        assert lon.is_contiguous()
        assert lat.is_contiguous()
        # 2D coords must be AuxCoords, which do not have a circular attribute.
        circular = False
    lon_bound_array = lon.contiguous_bounds()
    lon_bound_array = lon.units.convert(lon_bound_array, Unit("degrees"))
    lat_bound_array = lat.contiguous_bounds()
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
        )
    else:
        grid_info = RefinedGridInfo(
            lon_bound_array,
            lat_bound_array,
            crs=crs,
            resolution=resolution,
        )
    return grid_info


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
        grid_dim_x, grid_dim_y = src_dims
    elif len(src_dims) == 1:
        grid_dim_y = src_dims[0]
        grid_dim_x = grid_dim_y + 1
    else:
        raise ValueError(
            f"Source grid must be described by 1 or 2 dimensions, got {len(src_dims)}"
        )
    if num_tgt_dims == 1:
        grid_dim_x = grid_dim_y = min(src_dims)
    for tgt_coord, dim in zip(tgt_coords, (grid_dim_x, grid_dim_y)):
        if len(tgt_coord.shape) == 1:
            if isinstance(tgt_coord, iris.coords.DimCoord):
                new_cube.add_dim_coord(tgt_coord, dim)
            else:
                new_cube.add_aux_coord(tgt_coord, dim)
        else:
            new_cube.add_aux_coord(tgt_coord, (grid_dim_y, grid_dim_x))

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


def _regrid_rectilinear_to_rectilinear__prepare(src_grid_cube, tgt_grid_cube):
    tgt_x = tgt_grid_cube.coord(axis="x")
    tgt_y = tgt_grid_cube.coord(axis="y")
    src_x = src_grid_cube.coord(axis="x")
    src_y = src_grid_cube.coord(axis="y")

    if len(src_x.shape) == 1:
        grid_x_dim = src_grid_cube.coord_dims(src_x)[0]
        grid_y_dim = src_grid_cube.coord_dims(src_y)[0]
    else:
        grid_y_dim, grid_x_dim = src_grid_cube.coord_dims(src_x)

    srcinfo = _cube_to_GridInfo(src_grid_cube)
    tgtinfo = _cube_to_GridInfo(tgt_grid_cube)

    regridder = Regridder(srcinfo, tgtinfo)

    regrid_info = RegridInfo(
        x_dim=grid_x_dim,
        y_dim=grid_y_dim,
        x_coord=tgt_x,
        y_coord=tgt_y,
        regridder=regridder,
    )

    return regrid_info


def _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    grid_x_dim, grid_y_dim, grid_x, grid_y, regridder = regrid_info

    # Set up a function which can accept just chunk of data as an argument.
    regrid = functools.partial(
        _regrid_along_grid_dims,
        regridder,
        grid_x_dim=grid_x_dim,
        grid_y_dim=grid_y_dim,
        mdtol=mdtol,
    )

    # Apply regrid to all the chunks of src_cube, ensuring first that all
    # chunks cover the entire horizontal plane (otherwise they would break
    # the regrid function).
    if len(grid_x.shape) == 1:
        chunk_shape = (len(grid_x.points), len(grid_y.points))
    else:
        chunk_shape = grid_x.shape
    new_data = map_complete_blocks(
        src_cube,
        regrid,
        (grid_x_dim, grid_y_dim),
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


def regrid_rectilinear_to_rectilinear(src_cube, grid_cube, mdtol=0):
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

    Returns
    -------
    :class:`iris.cube.Cube`
        A new :class:`~iris.cube.Cube` instance.

    """
    regrid_info = _regrid_rectilinear_to_rectilinear__prepare(src_cube, grid_cube)
    result = _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class ESMFAreaWeighted:
    """
    A scheme which can be recognised by :meth:`iris.cube.Cube.regrid`.

    This class describes an area-weighted regridding scheme for regridding
    between horizontal grids with separated ``X`` and ``Y`` coordinates. It uses
    :mod:`ESMF` to be able to handle grids in different coordinate systems.
    """

    def __init__(self, mdtol=0):
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

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

    def __repr__(self):
        """Return a representation of the class."""
        return "ESMFAreaWeighted(mdtol={})".format(self.mdtol)

    def regridder(self, src_grid, tgt_grid):
        """
        Create regridder to perform regridding from ``src_grid`` to ``tgt_grid``.

        Parameters
        ----------
        src_grid : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` defining the source grid.
        tgt_grid : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns
        -------
        :class:`ESMFAreaWeightedRegridder`
            A callable instance of a regridder with the interface: ``regridder(cube)``
                ... where ``cube`` is a :class:`~iris.cube.Cube` with the same
                grid as ``src_grid`` that is to be regridded to the grid of
                ``tgt_grid``.
        """
        return ESMFAreaWeightedRegridder(src_grid, tgt_grid, mdtol=self.mdtol)


class ESMFAreaWeightedRegridder:
    r"""Regridder class for unstructured to rectilinear :class:`~iris.cube.Cube`\\ s."""

    def __init__(self, src_grid, tgt_grid, mdtol=0):
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

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        regrid_info = _regrid_rectilinear_to_rectilinear__prepare(src_grid, tgt_grid)

        # Store regrid info.
        self.grid_x = regrid_info.x_coord
        self.grid_y = regrid_info.y_coord
        self.regridder = regrid_info.regridder

        # Record the source grid.
        self.src_grid = (src_grid.coord(axis="x"), src_grid.coord(axis="y"))

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
        src_x, src_y = (cube.coord(axis="x"), cube.coord(axis="y"))

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
