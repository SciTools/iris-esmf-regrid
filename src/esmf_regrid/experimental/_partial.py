"""Provides a regridder class compatible with Partition."""

import numpy as np

from esmf_regrid.schemes import (
    _create_cube,
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

    def partial_regrid(self, src):
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
        dims = self._get_cube_dims(src_cube)

        result_data = self.regridder._regrid_from_weights_and_data(weights, data)
        result_cube = _create_cube(
            result_data, src_cube, dims, self._tgt, len(self._tgt)
        )
        return result_cube
