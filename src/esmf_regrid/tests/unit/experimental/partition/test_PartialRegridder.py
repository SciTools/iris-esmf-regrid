"""Unit tests for :mod:`esmf_regrid.experimental.partition`."""

from esmf_regrid import ESMFAreaWeighted
from esmf_regrid.experimental._partial import PartialRegridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _grid_cube,
)

def test_PartialRegridder_repr():
    """Test repr of PartialRegridder instance."""
    src = _grid_cube(10, 15, (-180, 180), (-90, 90), circular=True)
    tgt = _grid_cube(5, 10, (-180, 180), (-90, 90), circular=True)
    src_slice = ((10,20), (15, 30))
    tgt_slice = ((0, 5), (0, 10))
    weights = None
    scheme = ESMFAreaWeighted(mdtol=0.5)

    pr = PartialRegridder(src, tgt, src_slice, tgt_slice, weights, scheme)

    expected_repr = ("PartialRegridder(src_slice=((10, 20), (15, 30)), tgt_slice=((0, 5), (0, 10)), "
                     "scheme=ESMFAreaWeighted(mdtol=0.5, use_src_mask=False, use_tgt_mask=False, esmf_args={}))")
    assert repr(pr) == expected_repr
