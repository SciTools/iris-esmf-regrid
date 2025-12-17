"""Provides an interface for splitting up a large regridding task."""

import numpy as np

from esmf_regrid.constants import Constants
from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental._partial import PartialRegridder
from esmf_regrid.schemes import _get_grid_dims


def _get_chunk(cube, sl):
    if cube.mesh is None:
        grid_dims = _get_grid_dims(cube)
    else:
        grid_dims = (cube.mesh_dim(),)
    full_slice = [np.s_[:]] * len(cube.shape)
    for s, d in zip(sl, grid_dims):
        full_slice[d] = np.s_[s[0]:s[1]]
    return cube[*full_slice]

def _determine_blocks(shape, chunks, num_chunks, explicit_chunks):
    which_inputs = (chunks is not None, num_chunks is not None, explicit_chunks is not None)
    if sum(which_inputs) == 0:
        msg = "Partition blocks must must be specified by either chunks, num_chunks, or explicit_chunks."
        raise ValueError(msg)
    if sum(which_inputs) > 1:
        msg = "Potentially conflicting partition block definitions."
        raise ValueError(msg)
    if num_chunks is not None:
        chunks = [s//n for s, n in zip(shape, num_chunks)]
        for chunk in chunks:
            if chunk == 0:
                msg = "`num_chunks` cannot divide a dimension into more blocks than the size of that dimension."
                raise ValueError(msg)
    if chunks is not None:
        if all(isinstance(x, int)for x in chunks):
            proper_chunks = []
            for s, c in zip(shape, chunks):
                proper_chunk = [c] * (s//c)
                if s%c != 0:
                    proper_chunk += [s%c]
                proper_chunks.append(proper_chunk)
            chunks = proper_chunks
        for s, chunk in zip(shape, chunks):
            if sum(chunk) != s:
                msg = "Chunks must sum to the size of their respective dimension."
                raise ValueError(msg)
        bounds = [np.cumsum([0] + list(chunk)) for chunk in chunks]
        if len(bounds) == 1:
            explicit_chunks = [[[int(lower), int(upper)]] for lower, upper in zip(bounds[0][:-1], bounds[0][1:])]
        elif len(bounds) == 2:
            explicit_chunks = [
                [[int(ly), int(uy)], [int(lx), int(ux)]] for ly, uy in zip(bounds[0][:-1], bounds[0][1:]) for lx, ux in zip(bounds[1][:-1], bounds[1][1:])
            ]
        else:
            msg = "Chunks must not exceed two dimensions."
            raise ValueError(msg)
    return explicit_chunks

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
            explicit_src_chunks=None,
            # tgt_chunks=None,
            # num_tgt_chunks=None,
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
        scheme :
            scheme
        file_names : iterable of str
            A list of file names to save/load parts of the regridder to/from.
        use_dask_src_chunks : bool, default=False
            If true, partition using the same chunks from the source cube.
        src_chunks : numpy array, tuple of int or str, default=None
            Specify the size of blocks to use to divide up the cube. Demensions are specified
            in y,x axis order. If `src_chunks` is a tuple of int, each integer describes
            the maximum size of a block in that dimension. If `src_chunks` is a tuple of int,
            each tuple describes the size of each successive block in that dimension. These
            block sizes should add up to the total size of that dimension or else an error
            is raised.
        num_src_chunks : tuple of int
            Specify the number of blocks to use to divide up the cube. Demensions are specified
            in y,x axis order. Each integer describes the number of blocks that dimension will
            be divided into.
        explicit_src_chunks : arraylike NxMx2
            Explicitly specify the bounds of each block in the partition.
        # tgt_chunks : ???, default=None
        #     ???
        # num_tgt_chunks : tuple of int
        #     ???
        auto_generate : bool, default=False
            When true, start generating files on initialisation.
        saved_files : iterable of str
            A list of paths to previously saved files.
        """
        if scheme._method == Constants.Method.NEAREST:
            msg = "The `Nearest` method is not implemented."
            raise NotImplementedError(msg)
        if src.mesh is not None:
            msg = "Partition does not yet support source meshes."
            raise NotImplementedError(msg)
        # TODO Extract a slice of the cube.
        self.src = src
        if src.mesh is None:
            grid_dims = _get_grid_dims(src)
        else:
            grid_dims = (src.mesh_dim(),)
        shape = tuple(src.shape[i] for i in grid_dims)
        self.tgt = tgt
        self.scheme = scheme
        # TODO: consider abstracting away the idea of files
        self.file_names = file_names
        if use_dask_src_chunks:
            assert num_src_chunks is None and src_chunks is None
            assert src.has_lazy_data()
            src_chunks = src.slices(grid_dims).next().lazy_data().chunks
        self.src_chunks = _determine_blocks(shape, src_chunks, num_src_chunks, explicit_src_chunks)
        assert len(self.src_chunks) == len(file_names)
        # This will be controllable in future
        tgt_chunks = None
        self.tgt_chunks = tgt_chunks
        assert tgt_chunks is None  # We don't handle big targets currently

        # Note: this may need to become more sophisticated when both src and tgt are large
        self.file_chunk_dict = {file: chunk for file, chunk in zip(self.file_names, self.src_chunks)}

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
            f"src={self.src}, "
            f"tgt={self.tgt}, "
            f"scheme={self.scheme}, "
            f"num file_names={len(self.file_names)},"
            f"num saved_files={len(self.saved_files)})"
        )
        return result

    @property
    def unsaved_files(self):
        files = set(self.file_names) - set(self.saved_files)
        return [file for file in self.file_names if file in files]

    def generate_files(self, files_to_generate=None):
        if files_to_generate is None:
            files = self.unsaved_files
        else:
            assert isinstance(files_to_generate, int)
            files = self.unsaved_files[:files_to_generate]

        for file in files:
            src_chunk = self.file_chunk_dict[file]
            src = _get_chunk(self.src, src_chunk)
            tgt = self.tgt
            regridder = self.scheme.regridder(src, tgt)
            weights = regridder.regridder.weight_matrix
            regridder = PartialRegridder(src, self.tgt, src_chunk, None, weights, self.scheme)
            save_regridder(regridder, file, allow_partial=True)
            self.saved_files.append(file)

    def apply_regridders(self, cube, allow_incomplete=False):
        # for each target chunk, iterate through each associated regridder
        # for now, assume one target chunk
        if not allow_incomplete:
            assert len(self.unsaved_files) == 0
        current_result = None
        current_weights = None
        files = self.saved_files

        for file, chunk in zip(self.file_names, self.src_chunks):
            if file in files:
                next_regridder = load_regridder(file, allow_partial=True)
                # cube_chunk = cube[*_interpret_slice(chunk)]
                cube_chunk = _get_chunk(cube, chunk)
                next_weights, next_result = next_regridder.partial_regrid(cube_chunk)
                if current_weights is None:
                    current_weights = next_weights
                else:
                    current_weights += next_weights
                if current_result is None:
                    current_result = next_result
                else:
                    current_result += next_result

        return next_regridder.finish_regridding(cube_chunk, current_weights, current_result)
