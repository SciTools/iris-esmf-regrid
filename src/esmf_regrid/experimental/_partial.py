"""Provides a regridder class compatible with Partition."""

from esmf_regrid.schemes import (
    _ESMFRegridder,
)

class PartialRegridder(_ESMFRegridder):
    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice
        self.scheme = scheme
        # TODO: consider disallowing ESMFNearest (unless out of bounds can be made masked)

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
