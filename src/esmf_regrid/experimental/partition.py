"""Provides an interface for splitting up a large regridding task."""
import esmpy
import numpy as np

from esmf_regrid.constants import Constants
from esmf_regrid.experimental._partial import PartialRegridder
from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.schemes import _get_grid_dims


def _get_chunk(cube, sl):
    if cube.mesh is None:
        grid_dims = _get_grid_dims(cube)
    else:
        grid_dims = (cube.mesh_dim(),)
    full_slice = [np.s_[:]] * len(cube.shape)
    for s, d in zip(sl, grid_dims, strict=True):
        full_slice[d] = np.s_[s[0] : s[1]]
    return cube[*full_slice]


def _determine_blocks(shape, chunks, num_chunks, explicit_blocks):
    which_inputs = (
        chunks is not None,
        num_chunks is not None,
        explicit_blocks is not None,
    )
    if sum(which_inputs) == 0:
        msg = "Partition blocks must must be specified by either chunks, num_chunks, or explicit_chunks."
        raise ValueError(msg)
    if sum(which_inputs) > 1:
        msg = "Potentially conflicting partition block definitions."
        raise ValueError(msg)
    if num_chunks is not None:
        chunks = [s // n for s, n in zip(shape, num_chunks, strict=True)]
        for chunk in chunks:
            if chunk == 0:
                msg = "`num_chunks` cannot divide a dimension into more blocks than the size of that dimension."
                raise ValueError(msg)
    if chunks is not None:
        if all(isinstance(x, int) for x in chunks):
            proper_chunks = []
            for s, c in zip(shape, chunks, strict=True):
                proper_chunk = [c] * (s // c)
                if s % c != 0:
                    proper_chunk += [s % c]
                proper_chunks.append(proper_chunk)
            chunks = proper_chunks
        for s, chunk in zip(shape, chunks, strict=True):
            if sum(chunk) != s:
                msg = "Chunks must sum to the size of their respective dimension."
                raise ValueError(msg)
        bounds = [np.cumsum([0, *chunk]) for chunk in chunks]
        if len(bounds) == 1:
            msg = "Chunks must have exactly two dimensions."
            raise ValueError(msg)
            # TODO: This is currently blocked by the fact that slicing an Iris cube on its mesh dimension
            #  does not currently yield another cube with a mesh. When this is fixed, the following
            #  code can be uncommented and the noqa on the following line can be removed.
            # explicit_blocks = [
            #     [[int(lower), int(upper)]]
            #     for lower, upper in zip(bounds[0][:-1], bounds[0][1:], strict=True)
            # ]
        elif len(bounds) == 2:  # noqa: RET506
            explicit_blocks = [
                [[int(ly), int(uy)], [int(lx), int(ux)]]
                for ly, uy in zip(bounds[0][:-1], bounds[0][1:], strict=True)
                for lx, ux in zip(bounds[1][:-1], bounds[1][1:], strict=True)
            ]
        else:
            msg = "Chunks must not exceed two dimensions."
            raise ValueError(msg)
    if len(explicit_blocks[0]) != len(shape):
        msg = "Dimensionality of blocks does not match the number of dimensions."
        raise ValueError(msg)
    return explicit_blocks


class Partition:
    """Class for breaking down regridding into manageable chunks."""

    def __init__(
        self,
        src,
        tgt,
        scheme,
        file_names,
        use_dask_src_chunks=False,
        src_chunks=None,
        num_src_chunks=None,
        explicit_src_blocks=None,
        auto_generate=False,
        saved_files=None,
    ):
        """Class for breaking down regridding into manageable chunks.

        Note
        ----
        Currently, it is only possible to divide the source grid into chunks.
        Meshes are not yet supported as a source.

        Parameters
        ----------
        src : cube
            Source cube.
        tgt : cube
            Target cube.
        scheme : regridding scheme
            Regridding scheme to generate regridders, either ESMFAreaWeighted or ESMFBilinear.
        file_names : iterable of str
            A list of file names to save/load parts of the regridder to/from.
        use_dask_src_chunks : bool, default=False
            If true, partition using the same chunks from the source cube.
        src_chunks : numpy array, tuple of int or tuple of tuple of int, default=None
            Specify the size of blocks to use to divide up the cube. Dimensions are specified
            in y,x axis order. If `src_chunks` is a tuple of int, each integer describes
            the maximum size of a block in that dimension. If `src_chunks` is a tuple of tuples,
            each sub-tuple describes the size of each successive block in that dimension. The sum
            of these block sizes in each of the sub-tuples should add up to the total size of that
            dimension or else an error is raised.
        num_src_chunks : tuple of int
            Specify the number of blocks to use to divide up the cube. Dimensions are specified
            in y,x axis order. Each integer describes the number of blocks that dimension will
            be divided into.
        explicit_src_blocks : arraylike NxMx2
            Explicitly specify the bounds of each block in the partition. Describes N blocks
            along M dimensions with a pair of upper and lower bounds. The upper and lower bounds
            describe a slice of an array, e.g. the bounds (3, 6) describe the indices 3, 4, 5 in
            a particular dimension.
        auto_generate : bool, default=False
            When true, start generating files on initialisation.
        saved_files : iterable of str
            A list of paths to previously saved files.
        """
        if scheme._method == Constants.Method.NEAREST:
            msg = "The `Nearest` method is not implemented."
            raise NotImplementedError(msg)
        if scheme._method == Constants.Method.BILINEAR:
            pole_method = scheme.esmf_args.get("pole_method")
            if pole_method != esmpy.PoleMethod.NONE:
                msg = ("Bilinear regridding must have a `pole_method` of `esmpy.PoleMethod.NONE` in "
                       "the `esmf_args` in order for Partition to work.`")
                raise ValueError(msg)
        # TODO: Extract a slice of the cube.
        self.src = src
        if src.mesh is None:
            grid_dims = _get_grid_dims(src)
        else:
            msg = "Partition does not yet support source meshes."
            raise NotImplementedError(msg)
            # TODO: This is currently blocked by the fact that slicing an Iris cube on its mesh dimension
            #  does not currently yield another cube with a mesh. When this is fixed, the following
            #  code can be uncommented.
            # grid_dims = (src.mesh_dim(),)
        shape = tuple(src.shape[i] for i in grid_dims)
        self.tgt = tgt
        self.scheme = scheme
        # TODO: consider abstracting away the idea of files
        self.file_names = file_names
        if use_dask_src_chunks:
            if src_chunks is not None:
                msg = ("`src_chunks` and `use_dask_src_chunks` may provide conflicting"
                       "partition block definitions.")
                raise ValueError(msg)
            if not src.has_lazy_data():
                msg = "If `use_dask_src_chunks=True`, the source cube must be lazy."
                raise TypeError(msg)
            src_chunks = src.slices(grid_dims).next().lazy_data().chunks
        self.src_blocks = _determine_blocks(
            shape, src_chunks, num_src_chunks, explicit_src_blocks
        )
        if len(self.src_blocks) != len(file_names):
            msg = "Number of source blocks does not match number of file names."
            raise ValueError(msg)
        # This will be controllable in future
        tgt_blocks = None
        self.tgt_blocks = tgt_blocks
        if tgt_blocks is not None:
            msg = "Target chunking not yet implemented."
            raise NotImplementedError(msg)

        # Note: this may need to become more sophisticated when both src and tgt are large
        self.file_block_dict = dict(zip(self.file_names, self.src_blocks, strict=True))

        if saved_files is None:
            self.saved_files = []
        else:
            self.saved_files = saved_files
        if auto_generate:
            self.generate_files(self.file_names)

    def __repr__(self):
        """Return a representation of the class."""
        result = (
            f"Partition("
            f"src={self.src!r}, "
            f"tgt={self.tgt!r}, "
            f"scheme={self.scheme}, "
            f"num file_names={len(self.file_names)},"
            f"num saved_files={len(self.saved_files)})"
        )
        return result

    @property
    def unsaved_files(self):
        """List of files not yet generated."""
        return [file for file in self.file_names if file not in self.saved_files]

    def generate_files(self, files_to_generate=None):
        """Generate files with regridding information.

        Parameters
        ----------
        files_to_generate : int, default=None
            Specify the number of files to generate, default behaviour is to generate all files.
        """
        if files_to_generate is None:
            files = self.unsaved_files
        else:
            if not isinstance(files_to_generate, int):
                msg = "`files_to_generate` must be an integer."
                raise ValueError(msg)
            files = self.unsaved_files[:files_to_generate]

        for file in files:
            src_block = self.file_block_dict[file]
            src = _get_chunk(self.src, src_block)
            tgt = self.tgt
            regridder = self.scheme.regridder(src, tgt)
            weights = regridder.regridder.weight_matrix
            regridder = PartialRegridder(
                src, tgt, src_block, None, weights, self.scheme
            )
            save_regridder(regridder, file, allow_partial=True)
            self.saved_files.append(file)

    def apply_regridders(self, cube, allow_incomplete=False):
        """Apply the saved regridders to a cube.

        Parameters
        ----------
        allow_incomplete : bool, default=False
            If False, raise an error if not all files have been generated. If True, perform
            regridding using the files which have been generated.
        """
        # for each target chunk, iterate through each associated regridder
        # for now, assume one target chunk
        if len(self.saved_files) == 0:
            msg = "No files have been generated."
            raise OSError(msg)
        if not allow_incomplete and len(self.unsaved_files) != 0:
            msg = "Not all files have been generated."
            raise OSError(msg)
        current_result = None
        current_weights = None
        files = self.saved_files

        for file, chunk in zip(self.file_names, self.src_blocks, strict=True):
            if file in files:
                next_regridder = load_regridder(file, allow_partial=True)
                cube_chunk = _get_chunk(cube, chunk)
                next_weights, next_result, extra = next_regridder.partial_regrid(
                    cube_chunk
                )
                if current_weights is None:
                    current_weights = next_weights
                else:
                    current_weights += next_weights
                if current_result is None:
                    current_result = next_result
                else:
                    current_result += next_result

        # NOTE: the final "finish_regridding" operation could be performed using any one
        #  of the partial regridders,but the correct "corresponding" slice of the source
        #  must be passed.
        #  See :meth:`~esmf_regrid.experimental._partial.PartialRegridder.finish_regridding`.
        return next_regridder.finish_regridding(
            cube_chunk,  # matches *this* partial regridder
            current_weights,
            current_result,
            extra,  # should be *the same* for all the partial results
        )
