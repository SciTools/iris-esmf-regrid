"""Provides an interface for splitting up a large regridding task."""

import numpy as np
from scipy import sparse

from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental._partial import PartialRegridder


def _interpret_slice(sl):
    # return [slice(s) for s in sl]
    return np.s_[sl[0][0]:sl[0][1], sl[1][0]:sl[1][1]]

# TODO: consider if nearest is appropriate with this way of doing things


class Partition:

    # TODO: add a way to save the Partition object.
    # TODO: make a way to find out which files have been saved from the last session.

    ## hold a list of files
    ## hold a collection of source indices
    ## alternately hold a collection of chunk indices
    ## note which indices are fully loaded
    def __init__(self, src, tgt, scheme, file_names, src_chunks, tgt_chunks=None, auto_generate=False, saved_files=None, partially_saved=None):
        self.src = src
        self.tgt = tgt
        self.scheme = scheme
        # TODO: consider abstracting away the idea of files
        self.file_names = file_names
        # TODO: consider deriving this from self.src.lazy_data()
        self.src_chunks = src_chunks
        assert len(src_chunks) == len(file_names)
        self.tgt_chunks = tgt_chunks
        assert tgt_chunks is None  # We don't handle big targets currently

        # Note: this may need to become more sophisticated when both src and tgt are large
        self.file_chunk_dict = {file: chunk for file, chunk in zip(self.file_names, self.src_chunks)}

        self.neighbouring_files = self._find_neighbours()
        if saved_files is None:
            self.saved_files = []
        else:
            self.saved_files = saved_files
        if partially_saved is None:
            self.partially_saved_files = []
        else:
            self.partially_saved_files = partially_saved
        if auto_generate:
            self.generate_files(self.file_names)

    @property
    def unsaved_files(self):
        files = list(set(self.file_names) - set(self.saved_files) - set(self.partially_saved_files))
        return [file for file in self.file_names if file in files]

    def generate_files(self, files_to_generate=None):
        if files_to_generate is None:
            # TODO: consider adding logic to order the files more efficiently.
            files = self.partially_saved_files + self.unsaved_files
        else:
            assert isinstance(files_to_generate, int)
            files = (self.partially_saved_files + self.unsaved_files)[:files_to_generate]

        # sort files
        files = [file for file in self.file_names if file in files]

        # Do this to ensure the last regridder is saved
        files.append(None)

        previous_regridders = []
        previous_files = []
        for file in files:
            if file in self.partially_saved_files:
                next_regridder = load_regridder(file, allow_partial=True)
            elif file is None:
                next_regridder = None
            else:
                src_chunk = self.file_chunk_dict[file]
                src = self.src[*_interpret_slice(src_chunk)]
                tgt = self.tgt
                next_regridder = self.scheme.regridder(src, tgt)
            previous_regridders, next_regridder = self._combine_regridders(previous_regridders, next_regridder, previous_files, file)
            if previous_regridders:
                for regridder, pre_file in zip(previous_regridders, previous_files):
                    neighbours = self.neighbouring_files[pre_file]
                    file_complete = True
                    # TODO: consider any
                    for neighbour in neighbours:
                        if neighbour in self.unsaved_files and neighbour != file:
                            file_complete = False
                    save_regridder(regridder, pre_file, allow_partial=True)
                    if file_complete:
                        self.saved_files.append(pre_file)
                    else:
                        self.partially_saved_files.append(pre_file)

            # This will need to be more sophisticated for more complex cases
            previous_regridders = [next_regridder]
            previous_files = [file]


    def apply_regridders(self, cube, allow_incomplete=False):
        # for each target chunk, iterate through each associated regridder
        # for now, assume one target chunk
        # TODO: figure out how to mask parts of the target not covered by any source (e.g. start out with full mask)
        if not allow_incomplete:
            assert len(self.unsaved_files) == 0
        # TODO: this may work better as a cube of the correct shape for more complex cases
        current_result = None
        files = self.saved_files

        for file in files:
            # TODO: make sure this works well with dask
            next_regridder = load_regridder(file, allow_partial=True)
            cube_slice = next_regridder.src_slice
            next_result = next_regridder(cube[*_interpret_slice(cube_slice)])
            current_result = self._combine_results(current_result, next_result)
        return current_result


    def _find_neighbours(self):
        # for the simplest case, neighbours will be next to each other in the list
        files = self.file_names
        neighbours = {file_1: (file_0, file_2) for file_0, file_1, file_2 in zip(files[:-2], files[1:-1], files[2:])}
        neighbours.update({files[0]: (files[1],), files[-1]: (files[-2],)})
        return neighbours

    def _combine_regridders(self, existing, current_regridder, pre_files, file):
        # For now, combine 2, in future, more regridders may be combined

        if len(existing) == 0:
            if not isinstance(current_regridder, PartialRegridder):
                src_slice = self.file_chunk_dict[file]
                src_cube = self.src[*_interpret_slice(src_slice)]
                weights = current_regridder.regridder.weight_matrix
                current_regridder = PartialRegridder(src_cube, self.tgt, src_slice, None, weights, self.scheme)
        elif current_regridder is None:
            previous_regridder, = existing
            if not isinstance(previous_regridder, PartialRegridder):
                src_slice = self.file_chunk_dict[pre_files[0]]
                src_cube = self.src[*_interpret_slice(src_slice)]
                weights = previous_regridder.regridder.weight_matrix
                previous_regridder = PartialRegridder(src_cube, self.tgt, src_slice, None, weights, self.scheme)
                existing = [previous_regridder]
        else:
            (previous_regridder,) = existing
            current_range = current_regridder.regridder.weight_matrix.max(axis=1).nonzero()[0]
            previous_range = previous_regridder.regridder.weight_matrix.max(axis=1).nonzero()[0]
            mutual_overlaps = np.intersect1d(current_range, previous_range)

            if mutual_overlaps.shape != (0,):

                # Calculate a slice of the current chunk which contains all the overlapping source cells.
                overlaps_next = current_regridder.regridder.weight_matrix[mutual_overlaps].nonzero()[1]
                h_len = current_regridder._src[0].shape[0]
                v_len = current_regridder._src[1].shape[-1]
                # TODO: make this more rigorous
                tgt_size = np.prod(self.tgt.shape)
                buffer = (overlaps_next % v_len).max() + 1

                # Add this slice to the previous chunk.
                pre_file, = pre_files
                src_slice = self.file_chunk_dict[pre_file]
                # assumes slice has form [[x_start, x_stop], [y_start, y_stop]]
                # TODO: consider how this affects file_chunk_dict
                src_slice[0][1] += buffer
                new_src_cube = self.src[*_interpret_slice(src_slice)]
                tgt_slice = None  # should describe all valid target indices

                # Create weights for new regridder
                previous_wm = previous_regridder.regridder.weight_matrix
                current_wm = current_regridder.regridder.weight_matrix
                right_wm = sparse.csr_array((tgt_size, buffer * h_len))
                buffer_inds = sum(np.meshgrid(np.arange(buffer), np.arange(h_len) * v_len)).flatten()
                right_wm[mutual_overlaps] = current_wm[mutual_overlaps][:, buffer_inds]
                new_weight_matrix = _combine_sparse(previous_wm, right_wm, h_len, v_len, buffer, tgt_size)
                new_weight_matrix = sparse.csr_matrix(
                    new_weight_matrix)  # must be matrix for ier (likely to change in future)

                # Remove weights from current regridder which have been added to previous regridder.
                current_regridder.regridder.weight_matrix[mutual_overlaps] = 0

                # Construct replacement for previous regridder with new weights and source cube.
                previous_regridder = PartialRegridder(new_src_cube, self.tgt, src_slice, tgt_slice, new_weight_matrix, self.scheme)
                existing = [previous_regridder]
            else:
                if not isinstance(current_regridder, PartialRegridder):
                    src_slice = self.file_chunk_dict[file]
                    src_cube = self.src[*_interpret_slice(src_slice)]
                    weights = current_regridder.regridder.weight_matrix
                    current_regridder = PartialRegridder(src_cube, self.tgt, src_slice, None, weights, self.scheme)
                previous_regridder, = existing
                if not isinstance(previous_regridder, PartialRegridder):
                    src_slice = self.file_chunk_dict[pre_files[0]]
                    src_cube = self.src[*_interpret_slice(src_slice)]
                    weights = previous_regridder.regridder.weight_matrix
                    previous_regridder = PartialRegridder(src_cube, self.tgt, src_slice, None, weights, self.scheme)
                    existing = [previous_regridder]

        return existing, current_regridder

    def _combine_results(self, existing_results, next_result):
        # iterate through for each target chunk
        # for now, assume one target chunk
        if existing_results is None:
            combined_result = next_result
        else:
            # combined_result = existing_results + next_result
            combined_data = np.ma.filled(existing_results.data, 0) + np.ma.filled(next_result.data, 0)
            combined_mask = np.ma.getmaskarray(existing_results.data) & np.ma.getmaskarray(next_result.data)
            combined_result = existing_results.copy()
            combined_result.data = np.ma.array(combined_data, mask=combined_mask)
        return combined_result


def _combine_sparse(left, right, w, a, b, t):
    # TODO: make this more clear
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

class Partition2:
    def __init__(self, src, tgt, scheme, file_names, src_chunks, tgt_chunks=None, auto_generate=False, saved_files=None,
                 partially_saved=None):
        self.src = src
        self.tgt = tgt
        self.scheme = scheme
        # TODO: consider abstracting away the idea of files
        self.file_names = file_names
        # TODO: consider deriving this from self.src.lazy_data()
        self.src_chunks = src_chunks
        assert len(src_chunks) == len(file_names)
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

    @property
    def unsaved_files(self):
        files = set(self.file_names) - set(self.saved_files)
        return [file for file in self.file_names if file in files]

    def generate_files(self, files_to_generate=None):
        if files_to_generate is None:
            # TODO: consider adding logic to order the files more efficiently.
            files = self.unsaved_files
        else:
            assert isinstance(files_to_generate, int)
            files = self.unsaved_files[:files_to_generate]

        for file in files:
            src_chunk = self.file_chunk_dict[file]
            src = self.src[*_interpret_slice(src_chunk)]
            tgt = self.tgt
            regridder = self.scheme.regridder(src, tgt)
            src_slice = self.file_chunk_dict[file]
            src_cube = self.src[*_interpret_slice(src_slice)]
            print(src_cube)
            weights = regridder.regridder.weight_matrix
            regridder = PartialRegridder(src_cube, self.tgt, src_slice, None, weights, self.scheme)
            # TODO: make partial?
            save_regridder(regridder, file, allow_partial=True)
            self.saved_files.append(file)

    def apply_regridders(self, cube, allow_incomplete=False):
        # for each target chunk, iterate through each associated regridder
        # for now, assume one target chunk
        # TODO: figure out how to mask parts of the target not covered by any source (e.g. start out with full mask)
        if not allow_incomplete:
            assert len(self.unsaved_files) == 0
        # TODO: this may work better as a cube of the correct shape for more complex cases
        current_result = None
        current_weights = None
        files = self.saved_files

        for file, chunk in zip(self.file_names, self.src_chunks):
            if file in files:
                next_regridder = load_regridder(file, allow_partial=True)
                cube_chunk = cube[*_interpret_slice(chunk)]
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
