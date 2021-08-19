"""Provides representations of ESMF's Spatial Discretisation Objects."""

from abc import ABC, abstractmethod

import cartopy.crs as ccrs
import ESMF
import numpy as np


class SDO(ABC):
    """
    Abstract base class for handling spatial discretisation objects.

    This contains shared things for representing the three spatial discretisation
    objects supported by ESMPy, Grids, Meshes, and LocStreams.
    """

    def __init__(self, shape, index_offset, field_kwargs):
        self._shape = shape
        self._index_offset = index_offset
        self._field_kwargs = field_kwargs

    @abstractmethod
    def _make_esmf_sdo(self):
        pass

    def make_esmf_field(self):
        """Return an ESMF field representing the spatial discretisation object."""
        sdo = self._make_esmf_sdo()
        field = ESMF.Field(sdo, **self._field_kwargs)
        return field

    @property
    def shape(self):
        """Return shape."""
        return self._shape

    @property
    def dims(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        """Return the number of cells in the sdo."""
        return np.prod(self._shape)

    @property
    def index_offset(self):
        """Return the index offset."""
        return self._index_offset

    def _array_to_matrix(self, array):
        """
        Reshape data to a form that is compatible with weight matrices.

        The data should be presented in the form of a matrix (i.e. 2D) in order
        to be compatible with the weight matrix.
        Weight matrices deriving from ESMF use fortran ordering when flattening
        grids to determine cell indices so we use the same order for reshaping.
        We then take the transpose so that matrix multiplication happens over
        the appropriate axes.
        """
        return array.T.reshape((self.size, -1))

    def _matrix_to_array(self, array, extra_dims):
        """
        Reshape data to restore original dimensions.

        This is the inverse operation of `_array_to_matrix`.
        """
        return array.reshape((extra_dims + self._shape)[::-1]).T


class GridInfo(SDO):
    """
    Class for handling structured grids.

    This class holds information about lat-lon type grids. That is, grids
    defined by lists of latitude and longitude values for points/bounds
    (with respect to some coordinate reference system i.e. rotated pole).
    It contains methods for translating this information into ESMF objects.
    In particular, there are methods for representing as an ESMF Grid and
    as an ESMF Field containing that Grid. This ESMF Field is designed to
    contain enough information for area weighted regridding and may be
    inappropriate for other ESMF regridding schemes.

    """

    # TODO: Edit GridInfo so that it is able to handle 2D lat/lon arrays.

    def __init__(
        self,
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
        self.lons = lons
        self.lats = lats
        self.lonbounds = lonbounds
        self.latbounds = latbounds
        if crs is None:
            self.crs = ccrs.Geodetic()
        else:
            self.crs = crs
        self.circular = circular
        self.areas = areas
        super().__init__(
            shape=(len(lats), len(lons)),
            index_offset=1,
            field_kwargs={"staggerloc": ESMF.StaggerLoc.CENTER},
        )

    def _as_esmf_info(self):
        shape = np.array(self._shape)

        if self.circular:
            adjustedlonbounds = self.lonbounds[:-1]
        else:
            adjustedlonbounds = self.lonbounds

        centerlons, centerlats = np.meshgrid(self.lons, self.lats)
        cornerlons, cornerlats = np.meshgrid(adjustedlonbounds, self.latbounds)

        truecenters = ccrs.Geodetic().transform_points(self.crs, centerlons, centerlats)
        truecorners = ccrs.Geodetic().transform_points(self.crs, cornerlons, cornerlats)

        # The following note in xESMF suggests that the arrays passed to ESMPy ought to
        # be fortran ordered:
        # https://xesmf.readthedocs.io/en/latest/internal_api.html#xesmf.backend.warn_f_contiguous
        # It is yet to be determined what effect this has on performance.
        truecenterlons = np.asfortranarray(truecenters[..., 0])
        truecenterlats = np.asfortranarray(truecenters[..., 1])
        truecornerlons = np.asfortranarray(truecorners[..., 0])
        truecornerlats = np.asfortranarray(truecorners[..., 1])

        info = (
            shape,
            truecenterlons,
            truecenterlats,
            truecornerlons,
            truecornerlats,
            self.circular,
            self.areas,
        )
        return info

    def _make_esmf_sdo(self):
        info = self._as_esmf_info()
        (
            shape,
            truecenterlons,
            truecenterlats,
            truecornerlons,
            truecornerlats,
            circular,
            areas,
        ) = info

        if circular:
            grid = ESMF.Grid(
                shape,
                pole_kind=[1, 1],
                num_peri_dims=1,
                periodic_dim=1,
                pole_dim=0,
            )
        else:
            grid = ESMF.Grid(shape, pole_kind=[1, 1])

        grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_x = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_x[:] = truecornerlons
        grid_corner_y = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CORNER)
        grid_corner_y[:] = truecornerlats

        # Grid center points would be added here, this is not necessary for
        # conservative area weighted regridding
        # grid.add_coords(staggerloc=ESMF.StaggerLoc.CENTER)
        # grid_center_x = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CENTER)
        # grid_center_x[:] = truecenterlons
        # grid_center_y = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CENTER)
        # grid_center_y[:] = truecenterlats

        if areas is not None:
            grid.add_item(ESMF.GridItem.AREA, staggerloc=ESMF.StaggerLoc.CENTER)
            grid_areas = grid.get_item(
                ESMF.GridItem.AREA, staggerloc=ESMF.StaggerLoc.CENTER
            )
            grid_areas[:] = areas.T

        return grid
