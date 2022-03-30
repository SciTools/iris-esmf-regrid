"""Provides representations of ESMF's Spatial Discretisation Objects."""

from abc import ABC, abstractmethod

import cartopy.crs as ccrs
import ESMF
import numpy as np
import scipy.sparse


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
    def _refined_shape(self):
        """Return shape passed to ESMF."""
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
    def _refined_size(self):
        """Return the number of cells passed to ESMF."""
        return np.prod(self._refined_shape)

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
    It contains methods for translating this information into :mod:`ESMF` objects.
    In particular, there are methods for representing as a
    :class:`ESMF.api.grid.Grid` and
    as a :class:`ESMF.api.field.Field` containing that
    :class:`~ESMF.api.grid.Grid`. This ESMF :class:`~ESMF.api.field.Field`
    is designed to
    contain enough information for area weighted regridding and may be
    inappropriate for other :mod:`ESMF` regridding schemes.

    """

    def __init__(
        self,
        lons,
        lats,
        lonbounds,
        latbounds,
        crs=None,
        circular=False,
        areas=None,
        center=False,
    ):
        """
        Create a :class:`GridInfo` object describing the grid.

        Parameters
        ----------
        lons : :obj:`~numpy.typing.ArrayLike`
            A 1D or 2D array or list describing the longitudes of the
            grid points.
        lats : :obj:`~numpy.typing.ArrayLike`
            A 1D or 2D array or list describing the latitudes of the
            grid points.
        lonbounds : :obj:`~numpy.typing.ArrayLike`
            A 1D or 2D array or list describing the longitude bounds of
            the grid. Should have length one greater than ``lons``.
        latbounds : :obj:`~numpy.typing.ArrayLike`
            A 1D or 2D array or list describing the latitude bounds of
            the grid. Should have length one greater than ``lats``.
        crs : :class:`cartopy.crs.CRS`, optional
            Describes how to interpret the
            above arguments. If ``None``, defaults to :class:`~cartopy.crs.Geodetic`.
        circular : bool, default=False
            Describes if the final longitude bounds should
            be considered contiguous with the first.
        areas : :obj:`~numpy.typing.ArrayLike`, optional
            Array describing the areas associated with
            each face. If ``None``, then :mod:`ESMF` will use its own
            calculated areas.
        center : bool, default=False
            Describes if the center points of the grid cells are used in regridding
            calculations.

        """
        self.lons = lons
        self.lats = lats
        londims = len(self.lons.shape)
        if len(lonbounds.shape) != londims:
            msg = (
                f"The dimensionality of longitude bounds "
                f"({len(lonbounds.shape)}) is incompatible with the "
                f"dimensionality of the longitude ({londims})."
            )
            raise ValueError(msg)
        latdims = len(self.lats.shape)
        if len(latbounds.shape) != latdims:
            msg = (
                f"The dimensionality of latitude bounds "
                f"({len(latbounds.shape)}) is incompatible with the "
                f"dimensionality of the latitude ({latdims})."
            )
            raise ValueError(msg)
        if londims != latdims:
            msg = (
                f"The dimensionality of the longitude "
                f"({londims}) is incompatible with the "
                f"dimensionality of the latitude ({latdims})."
            )
            raise ValueError(msg)
        if londims not in (1, 2):
            msg = (
                f"Expected a latitude/longitude with a dimensionality "
                f"of 1 or 2, got {londims}."
            )
            raise ValueError(msg)
        if londims == 1:
            shape = (len(lats), len(lons))
        else:
            shape = self.lons.shape

        self.lonbounds = lonbounds
        self._refined_lonbounds = lonbounds
        self.latbounds = latbounds
        self._refined_latbounds = latbounds
        if crs is None:
            self.crs = ccrs.Geodetic()
        else:
            self.crs = crs
        self.circular = circular
        self.areas = areas
        self.center = center
        super().__init__(
            shape=shape,
            index_offset=1,
            field_kwargs={"staggerloc": ESMF.StaggerLoc.CENTER},
        )

    def _as_esmf_info(self):
        shape = np.array(self._refined_shape)

        londims = len(self.lons.shape)

        if londims == 1:
            if self.circular:
                adjustedlonbounds = self._refined_lonbounds[:-1]
            else:
                adjustedlonbounds = self._refined_lonbounds
            centerlons, centerlats = np.meshgrid(self.lons, self.lats)
            cornerlons, cornerlats = np.meshgrid(
                adjustedlonbounds, self._refined_latbounds
            )
        elif londims == 2:
            if self.circular:
                slice = np.s_[:, :-1]
            else:
                slice = np.s_[:]
            centerlons = self.lons[slice]
            centerlats = self.lats[slice]
            cornerlons = self._refined_lonbounds[slice]
            cornerlats = self._refined_latbounds[slice]

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

        # Grid center points are added here, this is not necessary for
        # conservative area weighted regridding
        if self.center:
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

        return grid


class RefinedGridInfo(GridInfo):
    """
    Class for handling structured grids represented in :mod:`ESMF` in higher resolution.

    A specialised version of :class:`GridInfo`. Designed to provide higher
    accuracy conservative regridding for rectilinear grids, especially those with
    particularly large cells which may not be well represented by :mod:`ESMF`. This
    class differs from :class:`GridInfo` primarily in the way it represents itself
    as a :class:`~ESMF.api.field.Field` in :mod:`ESMF`. This :class:`~ESMF.api.field.Field`
    is designed to be a higher resolution version of the given grid and should
    contain enough information for area weighted regridding but may be
    inappropriate for other :mod:`ESMF` regridding schemes.

    """

    def __init__(
        self,
        lonbounds,
        latbounds,
        resolution=3,
        crs=None,
    ):
        """
        Create a :class:`RefinedGridInfo` object describing the grid.

        Parameters
        ----------
        lonbounds : :obj:`~numpy.typing.ArrayLike`
            A 1D array or list describing the longitude bounds of the grid.
            Must be strictly increasing (for example, if a bound goes from
            170 to -170 consider transposing -170 to 190).
        latbounds : :obj:`~numpy.typing.ArrayLike`
            A 1D array or list describing the latitude bounds of the grid.
            Must be strictly increasing.
        resolution : int, default=400
            A number describing how many latitude slices each cell should
            be divided into when passing a higher resolution grid to ESMF.
        crs : :class:`cartopy.crs.CRS`, optional
            Describes how to interpret the
            above arguments. If ``None``, defaults to :class:`~cartopy.crs.Geodetic`.

        """
        # Convert bounds to numpy arrays where necessary.
        if not isinstance(lonbounds, np.ndarray):
            lonbounds = np.array(lonbounds)
        if not isinstance(latbounds, np.ndarray):
            latbounds = np.array(latbounds)

        # Ensure bounds are strictly increasing.
        if not np.all(lonbounds[:-1] < lonbounds[1:]):
            raise ValueError("The longitude bounds must be strictly increasing.")
        if not np.all(latbounds[:-1] < latbounds[1:]):
            raise ValueError("The latitude bounds must be strictly increasing.")

        self.resolution = resolution
        self.n_lons_orig = len(lonbounds) - 1
        self.n_lats_orig = len(latbounds) - 1

        # Create dummy lat/lon values
        lons = np.zeros(self.n_lons_orig)
        lats = np.zeros(self.n_lats_orig)
        super().__init__(lons, lats, lonbounds, latbounds, crs=crs)

        if self.n_lats_orig == 1 and np.allclose(latbounds, [-90, 90]):
            self._refined_latbounds = np.array([-90, 0, 90])
            self._refined_lonbounds = lonbounds
        else:
            self._refined_latbounds = latbounds
            self._refined_lonbounds = np.append(
                np.linspace(
                    lonbounds[:-1],
                    lonbounds[1:],
                    self.resolution,
                    endpoint=False,
                    axis=1,
                ).flatten(),
                lonbounds[-1],
            )
        self.lon_expansion = int(
            (len(self._refined_lonbounds) - 1) / (len(self.lonbounds) - 1)
        )
        self.lat_expansion = int(
            (len(self._refined_latbounds) - 1) / (len(self.latbounds) - 1)
        )

    @property
    def _refined_shape(self):
        """Return shape passed to ESMF."""
        return (
            self.n_lats_orig * self.lat_expansion,
            self.n_lons_orig * self.lon_expansion,
        )

    def _collapse_weights(self, is_tgt):
        """
        Return a matrix to collapse the weight matrix.

        The refined grid may contain more cells than the represented grid. When this is
        the case, the generated weight matrix will refer to too many points and will have
        to be collapsed. This is done by multiplying by this matrix, pre-multiplying when
        the target grid is represented and post multiplying when the source grid is
        represented.

        Parameters
        ----------
        is_tgt : bool
            True if the target field is being represented, False otherwise.
        """
        # The column indices represent each of the cells in the refined grid.
        column_indices = np.arange(self._refined_size)

        # The row indices represent the cells of the unrefined grid. These are broadcast
        # so that each row index coincides with all column indices of the refined cells
        # which the unrefined cell is split into.
        if self.lat_expansion > 1:
            # The latitudes are expanded only in the case where there is one latitude
            # bound from -90 to 90. In this case, there is no longitude expansion.
            row_indices = np.empty([self.n_lons_orig, self.lat_expansion])
            row_indices[:] = np.arange(self.n_lons_orig)[:, np.newaxis]
        else:
            # The row indices are broadcast across a dimension representing the expansion
            # of the longitude. Each row index is broadcast and flattened so that all the
            # row indices representing the unrefined cell match up with the column indices
            # representing the refined cells it is split into.
            row_indices = np.empty(
                [self.n_lons_orig, self.lon_expansion, self.n_lats_orig]
            )
            row_indices[:] = np.arange(self.n_lons_orig * self.n_lats_orig).reshape(
                [self.n_lons_orig, self.n_lats_orig]
            )[:, np.newaxis, :]
        row_indices = row_indices.flatten()
        matrix_shape = (self.size, self._refined_size)
        refinement_weights = scipy.sparse.csr_matrix(
            (
                np.ones(self._refined_size),
                (row_indices, column_indices),
            ),
            shape=matrix_shape,
        )
        if is_tgt:
            # When the RefinedGridInfo is the target of the regridder, we want to take
            # the average of the weights of each refined target cell. This is because
            # these weights represent the proportion of area of the target cells which
            # is covered by a given source cell. Since the refined cells are divided in
            # such a way that they have equal area, the weights for the unrefined cells
            # can be reconstructed by taking an average. This is done via matrix
            # multiplication, with the returned matrix pre-multiplying the weight matrix
            # so that it operates on the rows of the weight matrix (representing the
            # target cells). At this point the returned matrix consists of ones, so we
            # divided by the number of refined cells per unrefined cell.
            refinement_weights = refinement_weights / (
                self.lon_expansion * self.lat_expansion
            )
        else:
            # When the RefinedGridInfo is the source of the regridder, we want to take
            # the sum of the weights of each refined target cell. This is because those
            # weights represent the proportion of the area of a given target cell which
            # is covered by each refined source cell. The total proportion covered by
            # each unrefined source cell is then the sum of the weights from each of its
            # refined cells. This sum is done by matrix multiplication, the returned
            # matrix post-multiplying the weight matrix so that it operates on the columns
            # of the weight matrix (representing the source cells). In order for the
            # post-multiplication to work, the returned matrix must be transposed.
            refinement_weights = refinement_weights.T
        return refinement_weights
