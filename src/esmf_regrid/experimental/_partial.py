"""Provides a regridder class compatible with Partition."""

import numpy as np

from esmf_regrid.schemes import (
    GridRecord,
    MeshRecord,
    _create_cube,
    _ESMFRegridder,
)


class PartialRegridder(_ESMFRegridder):
    """Regridder class designed for use in :class:`~esmf_regrid.experimental.Partition`."""

    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        """Create a regridder instance for a block of :class:`~esmf_regrid.experimental.Partition`.

        Parameters
        ----------
        src : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` providing the source.
        tgt : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The :class:`~iris.cube.Cube` or :class:`~iris.mesh.MeshXY` providing the target.
        src_slice : tuple
            The upper and lower bounds of the block taken from the original source from which the
            ``src`` was derived. In the form ((x_low, x_high), ...) where x_low and x_high are the
            upper and lower bounds of the slice (in the x dimension) taken from the original source.
            There are as many tuples of upper and lower bounds as there are horizontal dimensions in
            the source cube (currently this is always 2 as Meshes are not yet supported for sources).
        tgt_slice : tuple
            The upper and lower bounds of the block taken from the original target from which the
            ``tgt`` was derived. In the form ((x_low, x_high), ...) where x_low and x_high are the
            upper and lower bounds of the slice (in the x dimension) taken from the original target.
            There are as many tuples of upper and lower bounds as there are horizontal dimensions in
            the target cube.
        weights : :class:`scipy.sparse.spmatrix`
            The weights to use for regridding.
        scheme : :class:`~esmf_regrid.schemes.ESMFAreaWeighted` or :class:`~esmf_regrid.schemes.ESMFBilinear`
            The scheme used to construct the regridder.
        kwargs : dict
            Additional keyword arguments to pass to the `scheme`s regridder method.
        """
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice
        self.scheme = scheme

        self._regridder = scheme.regridder(
            src,
            tgt,
            precomputed_weights=weights,
            **kwargs,
        )
        self.__dict__.update(self._regridder.__dict__)

    def __repr__(self):
        """Return a representation of the class."""
        result = (
            f"PartialRegridder("
            f"src={self._src}, "
            f"tgt_slice={self._tgt}, "
            f"src_slice={self.src_slice}, "
            f"tgt_slice={self.tgt_slice}, "
            f"scheme={self.scheme})"
        )
        return result

    def partial_regrid(self, src):
        """Perform the first half of regridding, generating weights and data."""
        dims = self._get_cube_dims(src)
        num_dims = len(dims)
        standard_in_dims = [-1, -2][:num_dims]
        data = np.moveaxis(src.data, dims, standard_in_dims)
        result = self.regridder._gen_weights_and_data(data)
        return result

    def finish_regridding(self, src_cube, weights, data, extra):
        """Perform the second half of regridding, combining weights and data.

        This operation is used to process the combined results from all the partial
        regridders in a Partition.
        Since all the combined data is passed in, this operation can be done using
        *any one* of the individual PartialRegridders.
        However, the passed "src_cube" must be the "correct" slice of the source
        data cube, corresponding to the 'tgt_slice' slice params it was created with.
        It is also implicit that the 'extra' arg (additional dimensions) will be the
        same for all partial results.
        The `src_cube` provides coordinates for the non-horizontal dimensions of the
        result cube, matching the dimensions of the `data` array.
        For technical convenience, its *horizontal* coordinates need to match those
        of the 'src' reference cube provided in regridder creation (`self._src`).
        So, it must be the correct "corresponding slice" of the source cube.
        """
        old_dims = self._get_cube_dims(src_cube)

        result_data = self.regridder._regrid_from_weights_and_data(weights, data, extra)

        num_out_dims = self.regridder.tgt.dims
        num_dims = len(old_dims)
        standard_out_dims = [-1, -2][:num_out_dims]
        if num_dims == 2 and num_out_dims == 1:
            new_dims = [min(old_dims)]
        elif num_dims == 1 and num_out_dims == 2:
            new_dims = [old_dims[0] + 1, old_dims[0]]
        else:
            new_dims = old_dims

        result_data = np.moveaxis(result_data, standard_out_dims, new_dims)

        if isinstance(self._tgt, GridRecord):
            tgt_coords = self._tgt
            out_dims = 2
        elif isinstance(self._tgt, MeshRecord):
            tgt_coords = self._tgt.mesh.to_MeshCoords(self._tgt.location)
            out_dims = 1
        else:
            msg = "Unrecognised target information."
            raise TypeError(msg)

        result_cube = _create_cube(
            result_data, src_cube, old_dims, tgt_coords, out_dims
        )
        return result_cube
