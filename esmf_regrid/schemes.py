"""TBD: public module docstring."""

__all__ = [
    "ESMFAreaWeighted",
]


class ESMFAreaWeighted:
    """TBD: public class docstring."""

    def regridder(self, src_grid, tgt_grid):
        """TBD: public method docstring."""
        return _ESMFAreaWeightedRegridder(src_grid, tgt_grid)


class _ESMFAreaWeightedRegridder:
    def __init__(self, src_grid, tgt_grid):
        # TODO implement esmf regridder as an iris scheme.
        return
