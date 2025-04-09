"""Provides an interface for splitting up a large regridding task."""


from esmf_regrid.schemes import _ESMFRegridder

class PartialRegridder(_ESMFRegridder):
    def __init__(self):
        pass

    ## keep track of source indices and target indices
    ## share reference to full source and target

class Partition:

    ## hold a list of files
    ## hold a collection of source indices
    ## note which indices are fully loaded
    def __init__(self, src, tgt, scheme, file_names, **kwargs):
        self.src = src
        self.tgt = tgt
        self.scheme = scheme
        self.file_names = file_names

        self.saved_files = []

    @property
    def unsaved_files(self):
        return list(set(self.file_names) - set(self.saved_files))

    def generate_files(self, files_to_generate=None):
        pass

    def apply_regridders(self):
        pass