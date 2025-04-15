"""Provides an interface for splitting up a large regridding task."""

import numpy as np
from scipy import sparse

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.schemes import _ESMFRegridder

class PartialRegridder(_ESMFRegridder):
    def __init__(self, src, tgt, src_slice, tgt_slice, weights, scheme, **kwargs):
        self.src_slice = src_slice
        self.tgt_slice = tgt_slice
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

    ## keep track of source indices and target indices
    ## share reference to full source and target

    ## works the same except everything out of bounds is 0 rather than masked

class Partition:

    ## hold a list of files
    ## hold a collection of source indices
    ## alternately hold a collection of chunk indices
    ## note which indices are fully loaded
    def __init__(self, src, tgt, scheme, file_names, src_chunks, tgt_chunks=None, auto_generate=False):
        self.src = src
        self.tgt = tgt
        self.scheme = scheme
        self.file_names = file_names
        # TODO: consider deriving this from self.src.lazy_data()
        self.src_chunks = src_chunks
        assert len(src_chunks) == len(file_names)
        self.tgt_chunks = tgt_chunks
        assert tgt_chunks == None # We don't handle big targets currently

        # Note: this may need to become more sophisticated when both src and tgt are large
        self.file_chunk_dict = {file: chunk for file, chunk in zip(self.file_names, self.src_chunks)}

        self.neighbouring_files = self._find_neighbours()

        self.saved_files = []
        self.partially_saved_files = []
        # self._src_slice_indices = []
        if auto_generate:
            self.generate_files(self.file_names)

    @property
    def unsaved_files(self):
        return list(set(self.file_names) - set(self.saved_files) - set(self.partially_saved_files))

    def generate_files(self, files_to_generate=None):
        if files_to_generate is None:
            # TODO: consider adding logic to order the files more efficiently.
            files = self.partially_saved_files + self.unsaved_files
        else:
            assert isinstance(files_to_generate, int)
            files = (self.partially_saved_files + self.unsaved_files)[:files_to_generate]

        # Do this to ensure the last regridder is saved
        files.append(None)

        previous_regridders = []
        previous_files = []
        for file in files:
            if file in self.partially_saved_files:
                next_regridder = load_regridder(file)
            elif file is None:
                next_regridder = None
            else:
                src = self.file_chunk_dict[file]
                tgt = self.tgt
                next_regridder = self.scheme.regridder(src, tgt)
            previous_regridders, next_regridder = self._combine_regridders(previous_regridders, previous_files, next_regridder)
            if previous_regridders:
                for regridder, pre_file in zip(previous_regridders, previous_files):
                    neighbours = self.neighbouring_files[pre_file]
                    file_complete = True
                    for neighbour in neighbours:
                        if neighbour in self.unsaved_files:
                            file_complete = False
                    save_regridder(regridder, pre_file)
                    if file_complete:
                        self.saved_files.append(pre_file)
                        # self._src_slice_indices.append(regridder.src_slice)
                    else:
                        self.partially_saved_files.append(pre_file)

            # This will need to be more sophisticated for more complex cases
            previous_regridders = [next_regridder]
            previous_files = [file]


    def apply_regridders(self, cube):
        # for each target chunk, iterate through each associated regridder
        # for now, assume one target chunk
        assert len(self.unsaved_files) == 0
        current_result = self.tgt.copy()
        for file in self.saved_files:
            next_regridder = load_regridder(file)
            cube_slice = next_regridder.src_slice
            next_result = next_regridder(cube[cube_slice])
            current_result = self._combine_results(current_result, next_result)
        return current_result


    # def _src_slices(self):
    #     for indices in self._src_slice_indices:
    #         yield self.src[indices]

    def _find_neighbours(self):
        # for the simplest case, neighbours will be next to each other in the list
        files = self.file_names
        neighbours = {file_1: (file_0, file_2) for file_0, file_1, file_2 in zip(files[:-2], files[1:-1], files[2:])}
        neighbours.update({files[0]: (files[1]), files[-1]: (files[-2])})
        return neighbours

    # def _determine_mask(self):
    #     assert len(self.unsaved_files) == 0
    #     # TODO: firgure out a way to do this when the target is big
    #     tgt_mask = np.ma.zeros_like(self.tgt.data)
    #     # TODO: calculate this mask
    #     return tgt_mask

    def _combine_regridders(self, existing, pre_files, current_regridder):
        # For now, combine 2, in future, more regridders may be combined
        if len(existing) == 0 or current_regridder is None:
            # TODO: turn these into partial regridders.
            return existing, current_regridder
        else:
            (previous_regridder,) = existing
            current_range = current_regridder.regridder.weight_matrix.max(axis=1).nonzero()[0]
            previous_range = previous_regridder.regridder.weight_matrix.max(axis=1).nonzero()[0]
            mutual_overlaps = np.intersect1d(current_range, previous_range)

            if mutual_overlaps.shape != (0,):

                # Calculate a slice of the current chunk which contains all the overlapping source cells.
                overlaps_next = current_regridder.regridder.weight_matrix[mutual_overlaps].nonzero()[1]
                h_len = current_regridder._src["grid_x"].shape[0]
                v_len = current_regridder._src["grid_y"].shape[-1]
                # TODO: make this more rigorous
                tgt_size = self.tgt.size
                buffer = (overlaps_next % h_len).max() + 1

                # Add this slice to the previous chunk.
                # TODO: make this refer to slices
                # new_src_cube = iris.cube.CubeList([previous_cube, cube[:buffer]]).concatenate_cube()
                # new_cubes.append(new_src_cube)
                pre_file, = pre_files
                src_slice = self.file_chunk_dict[pre_file]
                # assumes slice has form [[x_start, x_stop], [y_start, y_stop]]
                src_slice[0][1] += buffer
                # TODO: make this work
                new_src_cube = self.src[src_slice]
                tgt_slice = None

                # Create weights for new regridder
                previous_wm = previous_regridder.regridder.weight_matrix
                current_wm = current_regridder.regridder.weight_matrix
                right_wm = sparse.csr_array((tgt_size, buffer * v_len))
                buffer_inds = sum(np.meshgrid(np.arange(buffer), np.arange(v_len) * h_len)).flatten()
                right_wm[mutual_overlaps] = current_wm[mutual_overlaps][:, buffer_inds]
                new_weight_matrix = _combine_sparse(previous_wm, right_wm, v_len, h_len, buffer, tgt_size)
                new_weight_matrix = sparse.csr_matrix(
                    new_weight_matrix)  # must be matrix for ier (likely to change in future)

                # Remove weights from current regridder which have been added to previous regridder.
                current_regridder.regridder.weight_matrix[mutual_overlaps] = 0

                # Construct replacement for previous regridder with new weights and source cube.
                # previous_regridder = ESMFAreaWeightedRegridder(new_src_cube, tgt_mesh,
                #                                                precomputed_weights=new_weight_matrix)
                # previous_regridder.regridder.esmf_version = 0  # Must be set to allow regridder to load/save
                previous_regridder = PartialRegridder(new_src_cube, self.tgt, src_slice, tgt_slice, new_weight_matrix, self.scheme)
            else:
                current_regridder = PartialRegridder()
            # add combine code here
            pass

    def _combine_results(self, existing_results, next_result):
        # iterate through for each target chunk
        # for now, assume one target chunk
        return existing_results + next_result


def _combine_sparse(left, right, w, a, b, t):
    result = sparse.csr_array((t, w * (a + b)))
    src_indices_left = (np.arange(a)[np.newaxis, :] + ((a + b) * np.arange(w)[:, np.newaxis])).flatten()
    left_im = sparse.csr_array((np.ones(a * w), (np.arange(a * w), src_indices_left)), shape=(a * w, w * (a + b)))
    src_indices_right = (np.arange(b)[np.newaxis, :] + a + ((a + b) * np.arange(w)[:, np.newaxis])).flatten()
    right_im = sparse.csr_array((np.ones(b * w), (np.arange(b * w), src_indices_right)), shape=(b * w, w * (a + b)))

    result_add_left = left @ left_im
    result += result_add_left

    result_add_right = right @ right_im
    result += result_add_right
    return result

