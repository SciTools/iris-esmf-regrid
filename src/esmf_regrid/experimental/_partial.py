"""Provides a """

import numpy as np

from esmf_regrid.schemes import _ESMFRegridder
class PartialRegridder(_ESMFRegridder):
    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice  # this will be a boolean array
        self.scheme = scheme

        super().__init__(
            src,
            tgt,
            scheme.method,
            precomputed_weights=weights,
            **kwargs,
        )

    def __call__(self, cube):
        result = super().__call__(cube)

        # set everything outside the extent to 0
        inverse_slice = np.ones_like(result.data, dtype=bool)
        inverse_slice[self.tgt_slice] = 0
        # TODO: make sure this works with lazy data
        dims = self._get_cube_dims(cube)
        slice_slice = [np.newaxis] * result.ndim
        for dim in dims:
            slice_slice[dim] = np.s_[:]
        broadcast_slice = np.broadcast_to(inverse_slice[*slice_slice], result.shape)
        result.data[broadcast_slice] = 0
        return result

    def _get_src_slice(self, cube):
        # TODO: write a method to handle cubes of different dimensionalities
        return self.src_slice

    ## keep track of source indices and target indices
    ## share reference to full source and target

    ## works the same except everything out of bounds is 0 rather than masked