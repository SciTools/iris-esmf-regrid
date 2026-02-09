"""Unit tests for :mod:`esmf_regrid.experimental.partition`."""

import numpy as np

from esmf_regrid import ESMFAreaWeighted
from esmf_regrid.experimental._partial import PartialRegridder
from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _grid_cube,
)


def test_PartialRegridder_repr():
    """Test repr of PartialRegridder instance."""
    src = _grid_cube(10, 15, (-180, 180), (-90, 90), circular=True)
    tgt = _grid_cube(5, 10, (-180, 180), (-90, 90), circular=True)
    src_slice = ((10, 20), (15, 30))
    tgt_slice = ((0, 5), (0, 10))
    weights = None
    scheme = ESMFAreaWeighted(mdtol=0.5)

    pr = PartialRegridder(src, tgt, src_slice, tgt_slice, weights, scheme)

    expected_repr = (
        "PartialRegridder(src=GridRecord("
        "grid_x=<DimCoord: longitude / (degrees)  [-162., -126., ..., 126., 162.]+bounds  shape(10,)>, "
        "grid_y=<DimCoord: latitude / (degrees)  [-84., -72., ..., 72., 84.]+bounds  shape(15,)>), "
        "tgt_slice=GridRecord(grid_x=<DimCoord: longitude / (degrees)  [-144., -72., 0., 72., 144.]+bounds  shape(5,)>, "
        "grid_y=<DimCoord: latitude / (degrees)  [-81., -63., ..., 63., 81.]+bounds  shape(10,)>), "
        "src_slice=((10, 20), (15, 30)), tgt_slice=((0, 5), (0, 10)), scheme=ESMFAreaWeighted(mdtol=0.5, "
        "use_src_mask=False, use_tgt_mask=False, esmf_args={}))"
    )
    assert repr(pr) == expected_repr


def test_PartialRegridder_roundtrip(tmp_path):
    """Test load/save for PartialRegridder instance."""
    src = _grid_cube(10, 15, (-180, 180), (-90, 90), circular=True)
    mask = np.zeros_like(src.data)
    mask[0, 0] = 1
    src.data = np.ma.array(src.data, mask=mask)
    tgt = _grid_cube(5, 10, (-180, 180), (-90, 90), circular=True)
    src_slice = [[10, 20], [15, 30]]
    tgt_slice = [[0, 5], [0, 10]]
    weights = None
    scheme = ESMFAreaWeighted(
        mdtol=0.5, use_src_mask=src.data.mask, esmf_args={"ignore_degenerate": True}
    )

    pr = PartialRegridder(src, tgt, src_slice, tgt_slice, weights, scheme)
    file = tmp_path / "partial.nc"

    save_regridder(pr, file, allow_partial=True)
    loaded_pr = load_regridder(file, allow_partial=True)

    assert repr(loaded_pr) == repr(pr)
