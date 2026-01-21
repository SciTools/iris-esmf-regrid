"""Provides a regridder class compatible with Partition."""

import numpy as np

from esmf_regrid.schemes import (
    _create_cube,
    _ESMFRegridder,
)


class PartialRegridder(_ESMFRegridder):
    """Regridder class designed for use in :class:`~esmf_regrid.experimental._partial.Partial`."""

    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        """Create a regridder instance for a block of :class:`~esmf_regrid.experimental._partial.Partial`.

        Parameters
        ----------
        src : :class:`iris.cube.Cube`
            The :class:`~iris.cube.Cube` providing the source.
        tgt : :class:`iris.cube.Cube` or :class:`iris.mesh.MeshXY`
            The :class:`~iris.cube.Cube` or :class:`~iris.mesh.MeshXY` providing the target.
        src_slice : tuple
            The upper and lower bounds of the block taken from the original source from which the
            ``src`` was derived.
        tgt_slice : tuple
            The upper and lower bounds of the block taken from the original target from which the
            ``tgt`` was derived.
        weights : :class:`scipy.sparse.spmatrix`
            The weights to use for regridding.
        scheme : :class:`~esmf_regrid.schemes.ESMFAreaWeighted` or :class:`~esmf_regrid.schemes.ESMFBilinear`
            The scheme used to construct the regridder.
        """
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice
        self.scheme = scheme

        # Pop duplicate kwargs.
        for arg in set(kwargs.keys()).intersection(vars(self.scheme)):
            kwargs.pop(arg)

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
            f"src_slice={self.src_slice}, "
            f"tgt_slice={self.tgt_slice}, "
            f"scheme={self.scheme})"
        )
        return result

    def partial_regrid(self, src):
        """Perform the first half of regridding, generating weights and data."""
        dims = self._get_cube_dims(src)
        num_out_dims = self.regridder.tgt.dims
        num_dims = len(dims)
        standard_in_dims = [-1, -2][:num_dims]
        data = np.moveaxis(src.data, dims, standard_in_dims)
        result = self.regridder._gen_weights_and_data(data)

        standard_out_dims = [-1, -2][:num_out_dims]
        if num_dims == 2 and num_out_dims == 1:
            dims = [min(dims)]
        if num_dims == 1 and num_out_dims == 2:
            dims = [dims[0] + 1, dims[0]]
        result = tuple(np.moveaxis(r, standard_out_dims, dims) for r in result)
        return result

    def finish_regridding(self, src_cube, weights, data):
        """Perform the second half of regridding, combining weights and data."""
        dims = self._get_cube_dims(src_cube)

        result_data = self.regridder._regrid_from_weights_and_data(weights, data)
        result_cube = _create_cube(
            result_data, src_cube, dims, self._tgt, len(self._tgt)
        )
        return result_cube
