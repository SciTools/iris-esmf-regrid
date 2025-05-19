"""Provides a regridder class compatible with Partition"""

from esmf_regrid.schemes import (
    _ESMFRegridder,
)

class PartialRegridder(_ESMFRegridder):
    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice
        self.scheme = scheme
        # TODO: consider disallowing ESMFNearest (unless out of bounds can be made masked)

        self._regridder = scheme.regridder(
            src,
            tgt,
            precomputed_weights=weights,
            # TODO: turn this back on
            # **kwargs,
        )
        self.__dict__.update(self._regridder.__dict__)

    def _get_src_slice(self, cube):
        # TODO: write a method to handle cubes of different dimensionalities
        return self.src_slice
