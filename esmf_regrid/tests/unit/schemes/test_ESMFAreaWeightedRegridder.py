"""Unit tests for :class:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

import numpy as np

from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from ._common_regridder import _CommonRegridder


class TestAreaWeighted(_CommonRegridder):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFAreaWeightedRegridder`."""

    REGRIDDER = ESMFAreaWeightedRegridder

    def test_masks(self):
        """Test that the `use_src_mask` and `use_tgt_mask` keywords work properly."""
        src_indexing = np.s_[:, 1:]
        tgt_indexing = np.s_[1:]
        super()._test_masks(src_indexing, tgt_indexing)
