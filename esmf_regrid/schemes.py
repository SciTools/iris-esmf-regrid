from .esmf_regridder import GridInfo, Regridder


class ESMFAreaWeighted:
    def regridder(self, src_grid, tgt_grid):
        return _ESMFAreaWeightedRegridder(src_grid, tgt_grid)


class _ESMFAreaWeightedRegridder:
    def __init__(self, src_grid, tgt_grid):
        # TODO implement esmf regridder as an iris scheme.
        return
