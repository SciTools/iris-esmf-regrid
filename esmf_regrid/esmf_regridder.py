"""Provides ESMF representations of grids/UGRID meshes and a modified regridder."""

import ESMF
import dask.array as da
import numpy as np
import sparse

from ._grid import GridInfo


__all__ = [
    "GridInfo",
    "Regridder",
]


def _get_regrid_weights(src_field, tgt_field):
    regridder = ESMF.Regrid(
        src_field,
        tgt_field,
        ignore_degenerate=True,
        regrid_method=ESMF.RegridMethod.CONSERVE,
        unmapped_action=ESMF.UnmappedAction.IGNORE,
        # Choosing the norm_type DSTAREA allows for mdtol type operations
        # to be performed using the weights information later on.
        norm_type=ESMF.NormType.DSTAREA,
        factors=True,
    )
    # Without specifying deep_copy=true, the information in weights
    # would be corrupted when the ESMF regridder is destoyed.
    weights = regridder.get_factors(deep_copy=True)
    # The weights contains all the information needed for regridding,
    # the ESMF objects can be safely removed.
    regridder.destroy()
    return weights


class Regridder:
    """TBD: public class docstring."""

    def __init__(self, src, tgt, precomputed_weights=None):
        """
        TBD: public method docstring summary (one line).

        Create a regridder designed to regrid data from a specified
        source mesh/grid to a specified target mesh/grid.

        Parameters
        ----------
        src : object
            A MeshInfo or GridInfo object describing the source mesh/grid.
            Data supplied to this regridder should be in a numpy array
            whose shape is compatible with src.
        tgt : object
            A MeshInfo or GridInfo oject describing the target mesh/grid.
            Data output by this regridder will be a numpy array whose
            shape is compatible with tgt.
        precomputed_weights : sparse-matix object, optional
            None or a scipy.sparse matrix. If None, ESMF will be used to
            calculate regridding weights. Otherwise, ESMF will be bypassed
            and precomputed_weights will be used as the regridding weights.
        """

        self.src_rank = src.rank
        self.src = src
        self.tgt = tgt

        if precomputed_weights is None:
            src_field = src.make_esmf_field()
            tgt_field = tgt.make_esmf_field()
            factors, factors_index = _get_regrid_weights(src_field, tgt_field)
            tensor_shape = src.shape + tgt.shape
            src_inds = np.unravel_index(factors_index[:, 0]-1, src.shape)
            tgt_inds = np.unravel_index(factors_index[:, 1]-1, tgt.shape)
            inds = np.vstack(src_inds + tgt_inds)
            weights = sparse.COO(inds, factors.astype('d'), shape=tensor_shape)
            self.weights = da.from_array(weights)
        else:
            if precomputed_weights.shape != self.tgt.shape + self.src.shape:
                msg = "Expected precomputed weights to have shape {}, got shape {} instead."
                raise ValueError(
                    msg.format(
                        self.tgt.shape + self.src.shape,
                        precomputed_weights.shape,
                    )
                )
            self.weights = precomputed_weights

    def _apply(self, src_array):
        return da.tensordot(src_array, self.weights, self.src_rank)

    def regrid(self, src_array, norm_type="fracarea", mdtol=1.):
        """
        Perform regridding on an array of data.

        Parameters
        ----------
        src_array : array_like
            A numpy array whose shape is compatible with self.src
        norm_type : string
            Either "fracarea" or "dstarea", defaults to "fracarea". Determines the
            type of normalisation applied to the weights. Normalisations correspond
            to ESMF constants ESMF.NormType.FRACAREA and ESMF.NormType.DSTAREA.
        mdtol : float, optional
            A number between 0 and 1 describing the missing data tolerance.
            Depending on the value of `mdtol`, if a cell in the target grid is not
            sufficiently covered by unmasked cells of the source grid, then it will
            be masked. An `mdtol` of 1 means that only target cells which are not
            covered at all will be masked, an `mdtol` of 0 means that all target
            cells that are not entirely covered will be masked, and an `mdtol` of
            0.5 means that all target cells that are less than half covered will
            be masked.

        Returns
        -------
        array_like
            A numpy array whose shape is compatible with self.tgt.

        """
        filled_src = da.ma.filled(src_array, 0.)
        tgt_array = self._apply(filled_src)

        weight_sums = self._apply(~da.ma.getmaskarray(src_array))
        # Set the minimum mdtol to be slightly higher than 0 to account for rounding
        # errors.
        mdtol = min(max(mdtol, 1.e-8), 1. - 1.e-8)
        tgt_mask = weight_sums < mdtol
        if norm_type == "fracarea":
            weight_sums[weight_sums == 0.] = 1.
            tgt_array /= weight_sums
        elif norm_type == "dstarea":
            pass
        else:
            raise ValueError(f'Normalisation type "{norm_type}" is not supported')

        result = da.ma.masked_array(tgt_array, tgt_mask)
        return result
