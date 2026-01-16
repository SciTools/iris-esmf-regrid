"""Provides ESMF representations of grids/UGRID meshes and a modified regridder."""

import numpy as np
from numpy import ma
import scipy.sparse
from scipy.sparse import safely_cast_index_arrays

import esmf_regrid
from esmf_regrid import Constants, check_method, check_norm

from . import esmpy
from ._esmf_sdo import GridInfo, RefinedGridInfo

__all__ = [
    "GridInfo",
    "RefinedGridInfo",
    "Regridder",
]


ESMF_NO_VERSION = "N/A"


def _get_regrid_weights_dict(src_field, tgt_field, regrid_method, esmf_args=None):
    if esmf_args is None:
        esmf_args = {}
    else:
        esmf_args = esmf_args.copy()
    # Provide default values
    if "ignore_degenerate" not in esmf_args:
        esmf_args["ignore_degenerate"] = True
    if "unmapped_action" not in esmf_args:
        esmf_args["unmapped_action"] = esmpy.UnmappedAction.IGNORE
    # The value, in array form, that ESMF should treat as an affirmative mask.
    expected_mask = np.array([True])
    regridder = esmpy.Regrid(
        src_field,
        tgt_field,
        regrid_method=regrid_method,
        # Choosing the norm_type DSTAREA allows for mdtol type operations
        # to be performed using the weights information later on.
        norm_type=esmpy.NormType.DSTAREA,
        src_mask_values=expected_mask,
        dst_mask_values=expected_mask,
        factors=True,
        **esmf_args,
    )
    # Without specifying deep_copy=true, the information in weights_dict
    # would be corrupted when the ESMF regridder is destroyed.
    weights_dict = regridder.get_weights_dict(deep_copy=True)
    # The weights_dict contains all the information needed for regridding,
    # the ESMF objects can be safely removed.
    regridder.destroy()
    return weights_dict


def _weights_dict_to_sparse_array(weights, shape, index_offsets):
    matrix = scipy.sparse.csr_matrix(
        (
            weights["weights"],
            (
                weights["row_dst"] - index_offsets[0],
                weights["col_src"] - index_offsets[1],
            ),
        ),
        shape=shape,
    )
    return matrix


class Regridder:
    """Regridder for directly interfacing with :mod:`esmpy`."""

    def __init__(
        self,
        src,
        tgt,
        method=Constants.Method.CONSERVATIVE,
        precomputed_weights=None,
        esmf_args=None,
    ):
        """Create a regridder from descriptions of horizontal grids/meshes.

        Weights will be calculated using :mod:`esmpy` and stored as a
        :class:`scipy.sparse.csr_matrix`
        for use in regridding. If precomputed weights are provided,
        these will be used instead of calculating via :mod:`esmpy`.

        Parameters
        ----------
        src : :class:`~esmf_regrid.experimental.unstructured_regrid.MeshInfo` or :class:`GridInfo`
            Describes the source mesh/grid.
            Data supplied to this regridder should be in a :class:`numpy.ndarray`
            whose shape is compatible with ``src``.
        tgt : :class:`~esmf_regrid.experimental.unstructured_regrid.MeshInfo` or :class:`GridInfo`
            Describes the target mesh/grid.
            Data output by this regridder will be a :class:`numpy.ndarray` whose
            shape is compatible with ``tgt``.
        method : :class:`Constants.Method`
            The method to be used to calculate weights.
        precomputed_weights : :class:`scipy.sparse.spmatrix`, optional
            If ``None``, :mod:`esmpy` will be used to
            calculate regridding weights. Otherwise, :mod:`esmpy` will be bypassed
            and ``precomputed_weights`` will be used as the regridding weights.
        esmf_args : dict, optional
            A dictionary of arguments to pass to ESMF.
        """
        self.src = src
        self.tgt = tgt
        # check method is correct type
        self.method = check_method(method)

        self.esmf_regrid_version = esmf_regrid.__version__
        if precomputed_weights is None:
            self.esmf_version = esmpy.__version__
            src_field = src.make_esmf_field()
            src_sdo = src_field.grid
            tgt_field = tgt.make_esmf_field()
            tgt_sdo = tgt_field.grid
            try:
                weights_dict = _get_regrid_weights_dict(
                    src_field,
                    tgt_field,
                    regrid_method=method.value,
                    esmf_args=esmf_args,
                )
            finally:
                src_field.destroy()
                src_sdo.destroy()
                tgt_field.destroy()
                tgt_sdo.destroy()
            self.weight_matrix = _weights_dict_to_sparse_array(
                weights_dict,
                (self.tgt._refined_size, self.src._refined_size),
                (self.tgt.index_offset, self.src.index_offset),
            )
            if isinstance(tgt, RefinedGridInfo):
                # At this point, the weight matrix represents more target points than
                # tgt represents. In order to collapse these points, we collapse the
                # weights matrix by the appropriate matrix multiplication.
                self.weight_matrix = (
                    tgt._collapse_weights(is_tgt=True) @ self.weight_matrix
                )
            if isinstance(src, RefinedGridInfo):
                # At this point, the weight matrix represents more source points than
                # src represents. In order to collapse these points, we collapse the
                # weights matrix by the appropriate matrix multiplication.
                self.weight_matrix = self.weight_matrix @ src._collapse_weights(
                    is_tgt=False
                )
        else:
            if not scipy.sparse.issparse(precomputed_weights):
                e_msg = "Precomputed weights must be given as a sparse matrix."
                raise ValueError(e_msg)
            if precomputed_weights.shape != (self.tgt.size, self.src.size):
                msg = "Expected precomputed weights to have shape {}, got shape {} instead."
                raise ValueError(
                    msg.format(
                        (self.tgt.size, self.src.size),
                        precomputed_weights.shape,
                    )
                )
            self.esmf_version = ESMF_NO_VERSION
            self.weight_matrix = precomputed_weights

    def _out_dtype(self, in_dtype):
        """Return the expected output dtype for a given input dtype."""
        if self.method == Constants.Method.NEAREST:
            out_dtype = in_dtype
        else:
            weight_matrix = self.weight_matrix
            weight_dtype = weight_matrix.dtype
            out_dtype = (
                np.ones(1, dtype=in_dtype) * np.ones(1, dtype=weight_dtype)
            ).dtype
        return out_dtype

    def _gen_weights_and_data(self, src_array):
        extra_shape = src_array.shape[: -self.src.dims]

        if self.method == Constants.Method.NEAREST:
            weight_matrix = self.weight_matrix.astype(src_array.dtype)
        else:
            weight_matrix = self.weight_matrix

        flat_src = self.src._array_to_matrix(ma.filled(src_array, 0.0))
        flat_tgt = weight_matrix @ flat_src

        src_inverted_mask = self.src._array_to_matrix(~ma.getmaskarray(src_array))
        weight_sums = weight_matrix @ src_inverted_mask

        tgt_data = self.tgt._matrix_to_array(flat_tgt, extra_shape)
        tgt_weights = self.tgt._matrix_to_array(weight_sums, extra_shape)
        return tgt_weights, tgt_data

    def _regrid_from_weights_and_data(
        self, tgt_weights, tgt_data, norm_type=Constants.NormType.FRACAREA, mdtol=1
    ):
        # Set the minimum mdtol to be slightly higher than 0 to account for rounding
        # errors.
        mdtol = max(mdtol, 1e-8)
        tgt_mask = tgt_weights > 1 - mdtol
        normalisations = np.ones_like(tgt_data)
        if self.method != Constants.Method.NEAREST:
            masked_weight_sums = tgt_weights * tgt_mask
            if norm_type == Constants.NormType.FRACAREA:
                normalisations[tgt_mask] /= masked_weight_sums[tgt_mask]
            elif norm_type == Constants.NormType.DSTAREA:
                pass
        normalisations = ma.array(normalisations, mask=np.logical_not(tgt_mask))

        tgt_array = tgt_data * normalisations
        return tgt_array

    def regrid(self, src_array, norm_type=Constants.NormType.FRACAREA, mdtol=1):
        """Perform regridding on an array of data.

        Parameters
        ----------
        src_array : :obj:`~numpy.typing.ArrayLike`
            Array whose shape is compatible with ``self.src``.
        norm_type : :class:`Constants.NormType`
            Either ``Constants.NormType.FRACAREA`` or ``Constants.NormType.DSTAREA``.
            Determines the type of normalisation applied to the weights.
        mdtol : float, default=1
            A number between 0 and 1 describing the missing data tolerance.
            Depending on the value of ``mdtol``, if a cell in the target grid is not
            sufficiently covered by unmasked cells of the source grid, then it will
            be masked. ``mdtol=1`` means that only target cells which are not
            covered at all will be masked, ``mdtol=0`` means that all target
            cells that are not entirely covered will be masked, and ``mdtol=0.5``
            means that all target cells that are less than half covered will
            be masked.

        Returns
        -------
        :obj:`~numpy.typing.ArrayLike`
            An array whose shape is compatible with ``self.tgt``.

        """
        # Sets default value, as this can't be done with class attributes within method call
        norm_type = check_norm(norm_type)

        array_shape = src_array.shape
        main_shape = array_shape[-self.src.dims :]
        if main_shape != self.src.shape:
            e_msg = (
                f"Expected an array whose shape ends in {self.src.shape}, "
                f"got an array with shape ending in {main_shape}."
            )
            raise ValueError(e_msg)
        tgt_weights, tgt_data = self._gen_weights_and_data(src_array)
        tgt_array = self._regrid_from_weights_and_data(
            tgt_weights, tgt_data, norm_type=norm_type, mdtol=mdtol
        )
        return tgt_array
