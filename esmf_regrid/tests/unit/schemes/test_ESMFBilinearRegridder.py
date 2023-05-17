"""Unit tests for :class:`esmf_regrid.schemes.ESMFBilinearRegridder`."""

import numpy as np

from esmf_regrid.schemes import ESMFBilinearRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _grid_cube,
)
from ._common_regridder import _CommonRegridder


class TestBilinear(_CommonRegridder):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFBilinearRegridder`."""

    REGRIDDER = ESMFBilinearRegridder

    def test_masks(self):
        """Test that the `use_src_mask` and `use_tgt_mask` keywords work properly."""
        src_indexing = np.s_[2:]
        tgt_indexing = np.s_[2:]
        super()._test_masks(src_indexing, tgt_indexing)

    def test_regrid_data(self):
        """Test that regridding mathematics behaves in an expected way."""
        # Create two similar grids so that source data and expected result
        # data ought to look similar by visual inspection.
        src = _grid_cube(5, 4, [-180, 180], [-90, 90], circular=True)
        tgt = _grid_cube(4, 5, [-180, 180], [-90, 90], circular=True)

        src_data = np.arange(20).reshape([4, 5])
        src.data = src_data
        rg = self.REGRIDDER(src, tgt)

        expected_data = np.array(
            [
                [
                    0.3844461499074716,
                    1.4148972061736933,
                    2.585102793826307,
                    3.615553850092528,
                ],
                [
                    3.886915086072201,
                    5.423003230276512,
                    6.641000710423149,
                    7.588216702776281,
                ],
                [
                    7.649349191647961,
                    8.891001259926682,
                    10.108998740073318,
                    11.35065080835204,
                ],
                [
                    11.411783297223723,
                    12.358999289576854,
                    13.576996769723491,
                    15.1130849139278,
                ],
                [
                    15.384446149907474,
                    16.414897206173688,
                    17.58510279382631,
                    18.615553850092528,
                ],
            ]
        )
        result = rg(src)
        np.testing.assert_allclose(expected_data, result.data)
