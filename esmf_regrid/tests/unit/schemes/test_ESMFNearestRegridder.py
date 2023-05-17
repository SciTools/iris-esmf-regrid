"""Unit tests for :class:`esmf_regrid.schemes.ESMFNearestRegridder`."""

import numpy as np
import pytest

from esmf_regrid.schemes import ESMFNearestRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)
from ._common_regridder import _CommonRegridder


class TestNearest(_CommonRegridder):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFNearestRegridder`."""

    REGRIDDER = ESMFNearestRegridder

    # It should be noted that before this change this test was failing.
    # This is due to the fact that when two source points are equidistant from
    # a target point, rounding floating point differences due to unit
    # conversion would have an effect of the result.
    # These parameters have been tweaked so that no target point is equidistant
    # from two nearest source points.
    N_LONS_SRC = 5

    def test_masks(self):
        """Test that the `use_src_mask` and `use_tgt_mask` keywords work properly."""
        # Check other weights are correct. Note that unique to NEAREST_DTOS,
        # masking a source point causes the next nearest source point to gain
        # weights in the weight matrix. Because of this, we ignore the row
        # associated with that target point and check the rest of the weights
        # matrix.
        src_indexing = np.s_[1:]
        tgt_indexing = np.s_[1:]
        super()._test_masks(src_indexing, tgt_indexing)

    def test_invalid_mdtol(self):
        """Test erroring when mdtol is out of range - disabled for this subclass."""
        pytest.skip("mdtol inappropriate for Nearest scheme.")

    @pytest.mark.parametrize(
        "src_type,tgt_type",
        [
            ("grid", "grid"),
            ("grid", "mesh"),
            ("mesh", "grid"),
            ("grid", "curv"),
            ("curv", "grid"),
        ],
    )
    def test_regrid_data(self, src_type, tgt_type):
        """Test that regridding mathematics behaves in an expected way."""
        # Create two similar grids so that source data and expected result
        # data ought to look similar by visual inspection.
        if src_type == "grid":
            src = _grid_cube(5, 4, [-180, 180], [-90, 90], circular=True)
        elif src_type == "mesh":
            src = _gridlike_mesh_cube(5, 4)
        elif src_type == "curv":
            src = _curvilinear_cube(5, 4, [-180, 180], [-90, 90])

        if tgt_type == "grid":
            tgt = _grid_cube(4, 5, [-180, 180], [-90, 90], circular=True)
        elif tgt_type == "mesh":
            tgt = _gridlike_mesh_cube(4, 5)
        elif tgt_type == "curv":
            tgt = _curvilinear_cube(4, 5, [-180, 180], [-90, 90])

        if src_type == "mesh":
            src.data = np.arange(20)
        else:
            src_data = np.arange(20).reshape([4, 5])
            src.data = src_data

        rg = self.REGRIDDER(src, tgt)

        # when two source points are equidistant from a target point, the
        # chosen source point is dependent on the index which ESMF gives that
        # point. This decision is described by ESMF to be arbitrary, but
        # ought to be consistent when dealing with the same precise grid.
        expected_data = np.array(
            [
                [0, 1, 3, 4],
                [5, 6, 8, 9],
                [5, 6, 8, 9],
                [10, 11, 13, 14],
                [15, 16, 18, 19],
            ]
        )
        if tgt_type == "mesh":
            expected_data = expected_data.flatten()

        result = rg(src)
        np.testing.assert_allclose(expected_data, result.data)
