# -*- coding: utf-8 -*-

import cartopy.crs as ccrs
import ESMF
from iris.exceptions import CellMeasureNotFoundError
import numpy as np


def _bounds_cf_to_simple_1d(cf_bounds):
    assert (cf_bounds[1:, 0] == cf_bounds[:-1, 1]).all()
    simple_bounds = np.empty((cf_bounds.shape[0]+1,), dtype=np.float64)
    simple_bounds[:-1] = cf_bounds[:, 0]
    simple_bounds[-1] = cf_bounds[-1, 1]
    return simple_bounds


class GridInfo:
    """
    TBD: public class docstring summary (one line).

    This class holds information about lat-lon type grids. That is, grids
    defined by lists of latitude and longitude values for points/bounds
    (with respect to some coordinate reference system i.e. rotated pole).
    It contains methods for translating this information into ESMF objects.
    In particular, there are methods for representing as an ESMF Grid and
    as an ESMF Field containing that Grid. This ESMF Field is designed to
    contain enough information for area weighted regridding and may be
    inappropriate for other ESMF regridding schemes.

    """

    def __init__(self, grid, shape):
        self.grid = grid
        self.shape = shape

    @classmethod
    def from_cube(cls, cube):
        lat = cube.coord('latitude')
        lon = cube.coord('longitude')
        assert lat.ndim == lon.ndim
        ndim = lat.ndim
        assert lat.coord_system == lon.coord_system
        coord_sys = lat.coord_system
        if coord_sys:
            crs = coord_sys.as_cartopy_crs()
        else:
            crs = None
        if ndim == 1:
            try:
                areas = cube.cell_measure('cell_area')
            except CellMeasureNotFoundError:
                areas = None
            return GridInfo.from_1d_coords(
                lon.points,
                lat.points,
                _bounds_cf_to_simple_1d(lon.bounds),
                _bounds_cf_to_simple_1d(lat.bounds),
                crs,
                lon.circular,
                areas,
            )
        else:
            raise NotImplementedError

    @classmethod
    def from_1d_coords(
        cls,
        lons,
        lats,
        lonbounds,
        latbounds,
        crs=None,
        circular=False,
        areas=None,
    ):
        """
        Create a GridInfo object describing the grid.

        Parameters
        ----------
        lons : array_like
            A 1D numpy array or list describing the longitudes of the
            grid points.
        lats : array_like
            A 1D numpy array or list describing the latitudes of the
            grid points.
        lonbounds : array_like
            A 1D numpy array or list describing the longitude bounds of
            the grid. Should have length one greater than lons.
        latbounds : array_like
            A 1D numpy array or list describing the latitude bounds of
            the grid. Should have length one greater than lats.
        crs : cartopy projection, optional
            None or a cartopy.crs projection describing how to interpret the
            above arguments. If None, defaults to Geodetic().
        circular : bool, optional
            A boolean value describing if the final longitude bounds should
            be considered contiguous with the first. Defaults to False.
        areas : array_line, optional
            either None or a numpy array describing the areas associated with
            each face. If None, then ESMF will use its own calculated areas.

        """
        if crs is None:
            crs = ccrs.Geodetic()

        shape = (len(lats), len(lons))

        if circular:
            adjustedlonbounds = lonbounds[:-1]
        else:
            adjustedlonbounds = lonbounds

        centerlons, centerlats = np.meshgrid(lons, lats)
        cornerlons, cornerlats = np.meshgrid(adjustedlonbounds, latbounds)

        truecenters = ccrs.Geodetic().transform_points(crs, centerlons, centerlats)
        truecorners = ccrs.Geodetic().transform_points(crs, cornerlons, cornerlats)

        # The following note in xESMF suggests that the arrays passed to ESMPy ought to
        # be fortran ordered:
        # https://xesmf.readthedocs.io/en/latest/internal_api.html#xesmf.backend.warn_f_contiguous
        # It is yet to be determined what effect this has on performance.
        truecenterlons = np.asfortranarray(truecenters[..., 0].T)
        truecenterlats = np.asfortranarray(truecenters[..., 1].T)
        truecornerlons = np.asfortranarray(truecorners[..., 0].T)
        truecornerlats = np.asfortranarray(truecorners[..., 1].T)

        esmf_shape = np.array(tuple(reversed(shape)))
        if circular:
            num_peri_dims = 1
        else:
            num_peri_dims = 0
        grid = ESMF.Grid(esmf_shape, num_peri_dims=num_peri_dims, pole_kind=[1, 1])

        grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_x = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_x[:] = truecornerlons
        grid_corner_y = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_y[:] = truecornerlats

        grid.add_coords(staggerloc=ESMF.StaggerLoc.CENTER)
        grid_center_x = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CENTER)
        grid_center_x[:] = truecenterlons
        grid_center_y = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CENTER)
        grid_center_y[:] = truecenterlats

        if areas is not None:
            grid.add_item(ESMF.GridItem.AREA, staggerloc=ESMF.StaggerLoc.CENTER)
            grid_areas = grid.get_item(
                ESMF.GridItem.AREA, staggerloc=ESMF.StaggerLoc.CENTER
            )
            grid_areas[:] = areas.T

        return cls(grid, shape)

    def make_esmf_field(self):
        """TBD: public method docstring."""
        field = ESMF.Field(self.grid, staggerloc=ESMF.StaggerLoc.CENTER)
        return field

    def size(self):
        """TBD: public method docstring."""
        return np.prod(self.shape)

    def _index_offset(self):
        return 1
