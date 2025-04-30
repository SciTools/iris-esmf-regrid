"""Provides a regridder class compatible with Partition"""

from esmf_regrid.schemes import (
    _ESMFRegridder,
    # ESMFAreaWeighted,
    # ESMFAreaWeightedRegridder,
    # ESMFBilinear,
)

class PartialRegridder(_ESMFRegridder):
    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        self.src_slice = src_slice  # this will be tuple-like
        self.tgt_slice = tgt_slice
        self.scheme = scheme

        # kwargs = {
        #     "use_src_mask": scheme.use_src_mask,
        #     "use_tgt_mask": scheme.use_tgt_mask,
        #     "tgt_location": scheme.tgt_location,
        #     "esmf_args": scheme.esmf_args,
        # }
        # if isinstance(scheme, (ESMFAreaWeighted, ESMFBilinear)):
        #     kwargs["mdtol"] = scheme.mdtol


        # super().__init__(
        # ESMFAreaWeightedRegridder.__init__(
        #     src,
        #     tgt,
        #     scheme._method,
        #     precomputed_weights=weights,
        #     **kwargs,
        # )
        self._regridder = scheme.regridder(
            src,
            tgt,
            precomputed_weights=weights,
            # TODO: turn this back on
            # **kwargs,
        )
        self.__dict__.update(self._regridder.__dict__)

    def __call__(self, cube):
        result = super().__call__(cube)

        blank_cube = cube.copy()
        blank_cube.data[:] = 0
        result_mask = super().__call__(blank_cube).data.mask
        result.data[result_mask] = 0

        return result

    def _get_src_slice(self, cube):
        # TODO: write a method to handle cubes of different dimensionalities
        return self.src_slice

    ## keep track of source indices and target indices
    ## share reference to full source and target

    ## works the same except everything out of bounds is 0 rather than masked