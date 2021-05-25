"""Provides an iris interface for regridding."""

import copy

import iris
from iris.analysis._interpolation import get_xy_dim_coords
import numpy as np

from esmf_regrid.esmf_regridder import GridInfo, Regridder

__all__ = [
    "ESMFAreaWeighted",
    "ESMFAreaWeightedRegridder",
    "regrid_rectilinear_to_rectilinear",
]


def _bounds_cf_to_simple_1d(cf_bounds):
    assert (cf_bounds[1:, 0] == cf_bounds[:-1, 1]).all()
    simple_bounds = np.empty((cf_bounds.shape[0] + 1,), dtype=np.float64)
    simple_bounds[:-1] = cf_bounds[:, 0]
    simple_bounds[-1] = cf_bounds[-1, 1]
    return simple_bounds


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
    if cube.coord_system() is None:
        crs = None
    else:
        crs = cube.coord_system().as_cartopy_crs()
    return GridInfo(
        lon.points,
        lat.points,
        _bounds_cf_to_simple_1d(lon.bounds),
        _bounds_cf_to_simple_1d(lat.bounds),
        crs=crs,
        circular=lon.circular,
    )


def _regrid_along_grid_dims(regridder, data, grid_x_dim, grid_y_dim, mdtol):
    data = np.moveaxis(data, [grid_x_dim, grid_y_dim], [-1, -2])
    result = regridder.regrid(data, mdtol=mdtol)

    result = np.moveaxis(result, [-1, -2], [grid_x_dim, grid_y_dim])
    return result


def _create_cube(data, src_cube, grid_dim_x, grid_dim_y, grid_x, grid_y):
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
    grid_dim_x : int
        The dimension of the X coordinate within the source Cube.
    grid_dim_y : int
        The dimension of the Y coordinate within the source Cube.
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

    new_cube.add_dim_coord(grid_x, grid_dim_x)
    new_cube.add_dim_coord(grid_y, grid_dim_y)

    new_cube.metadata = copy.deepcopy(src_cube.metadata)

    def copy_coords(src_coords, add_method):
        for coord in src_coords:
            dims = src_cube.coord_dims(coord)
            if grid_dim_x in dims or grid_dim_y in dims:
                continue
            result_coord = coord.copy()
            # Add result_coord to the owner of add_method.
            add_method(result_coord, dims)

    copy_coords(src_cube.dim_coords, new_cube.add_dim_coord)
    copy_coords(src_cube.aux_coords, new_cube.add_aux_coord)

    return new_cube


def _regrid_rectilinear_to_rectilinear__prepare(src_grid_cube, tgt_grid_cube):
    tgt_x, tgt_y = get_xy_dim_coords(tgt_grid_cube)
    src_x, src_y = get_xy_dim_coords(src_grid_cube)

    grid_x_dim = src_grid_cube.coord_dims(src_x)[0]
    grid_y_dim = src_grid_cube.coord_dims(src_y)[0]

    srcinfo = _cube_to_GridInfo(src_grid_cube)
    tgtinfo = _cube_to_GridInfo(tgt_grid_cube)

    regridder = Regridder(srcinfo, tgtinfo)

    regrid_info = (grid_x_dim, grid_y_dim, tgt_x, tgt_y, regridder)

    return regrid_info


def _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    grid_x_dim, grid_y_dim, grid_x, grid_y, regridder = regrid_info

    # Perform regridding with realised data for the moment. This may be changed
    # in future to handle src_cube.lazy_data.
    new_data = _regrid_along_grid_dims(
        regridder, src_cube.data, grid_x_dim, grid_y_dim, mdtol
    )

    new_cube = _create_cube(
        new_data,
        src_cube,
        grid_x_dim,
        grid_y_dim,
        grid_x,
        grid_y,
    )
    return new_cube


def regrid_rectilinear_to_rectilinear(src_cube, grid_cube, mdtol=0):
    """
    Regrid rectilinear cube onto another rectilinear grid.

    Return a new cube with data values calculated using the area weighted
    mean of data values from the cube src_cube regridded onto the
    horizontal grid of grid_cube.

    Parameters
    ----------
    src_cube : cube
        An urectilinear instance of iris.cube.Cube that supplies the data,
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
    regrid_info = _regrid_rectilinear_to_rectilinear__prepare(src_cube, grid_cube)
    result = _regrid_rectilinear_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class ESMFAreaWeighted:
    """
    A scheme which can be recognised by an iris cube's regrid method.

    This class describes an area-weighted regridding scheme for regridding
    between horizontal grids with separated X and Y coordinates. It uses
    ESMF to be able to handle grids in different coordinate systems.
    """

    def __init__(self, mdtol=0):
        """
        Area-weighted scheme for regridding between rectilinear grids.

        Parameters
        ----------
        mdtol : float
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of missing data
            exceeds mdtol. This fraction is calculated based on the area of
            masked cells within each target cell. mdtol=0 means no masked
            data is tolerated while mdtol=1 will mean the resulting element
            will be masked if and only if all the overlapping elements of the
            source grid are masked. Defaults to 0.

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
        Create regridder to perform regridding from source grid to target grid.

        Parameters
        ----------
        src_grid : cube
            The cube defining the source grid.
        tgt_grid : cube
            The cube defining the target grid.

        Returns
        -------
        ESMFAreaWeightedRegridder
            A callable with the interface:
                `callable(cube)`
            where `cube` is a cube with the same grid as `src_grid`
            that is to be regridded to the grid of `tgt_grid`.
        """
        return ESMFAreaWeightedRegridder(src_grid, tgt_grid, mdtol=self.mdtol)


class ESMFAreaWeightedRegridder:
    """Regridder class for unstructured to rectilinear cubes."""

    def __init__(self, src_grid, tgt_grid, mdtol=0):
        """
        Create regridder for conversions between source grid and target grid.

        Parameters
        ----------
        src_grid_cube : cube
            The rectilinear iris cube providing the source grid.
        target_grid_cube : cube
            The rectilinear iris cube providing the target grid.
        mdtol : float, optional
            Tolerance of missing data. The value returned in each element of
            the returned array will be masked if the fraction of masked data
            exceeds mdtol. mdtol=0 means no missing data is tolerated while
            mdtol=1 will mean the resulting element will be masked if and only
            if all the contributing elements of data are masked.
            Defaults to 0.

        """
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        regrid_info = _regrid_rectilinear_to_rectilinear__prepare(src_grid, tgt_grid)

        # Store regrid info.
        _, _, self.grid_x, self.grid_y, self.regridder = regrid_info

        # Record the source grid.
        self.src_grid = get_xy_dim_coords(src_grid)

    def __call__(self, cube):
        """
        Regrid this cube onto the target grid of this regridder instance.

        The given cube must be defined with the same grid as the source
        cube used to create this ESMFAreaWeightedRegridder instance.

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
        src_x, src_y = get_xy_dim_coords(cube)

        # Check the source grid matches that used in initialisation
        assert self.src_grid == (src_x, src_y)

        grid_x_dim = cube.coord_dims(src_x)[0]
        grid_y_dim = cube.coord_dims(src_y)[0]

        regrid_info = (grid_x_dim, grid_y_dim, self.grid_x, self.grid_y, self.regridder)

        return _regrid_rectilinear_to_rectilinear__perform(
            cube, regrid_info, self.mdtol
        )
