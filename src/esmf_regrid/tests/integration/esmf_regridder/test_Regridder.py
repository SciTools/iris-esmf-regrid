"""Unit tests for :class:`esmf_regrid.esmf_regridder.Regridder`."""

import numpy as np
from numpy import ma

from esmf_regrid import Constants, esmpy
from esmf_regrid.esmf_regridder import GridInfo, Regridder
from esmf_regrid.tests import make_grid_args


def test_esmpy_normalisation():
    """Integration test for :meth:`~esmf_regrid.esmf_regridder.Regridder`.

    Checks against ESMF to ensure results are consistent.
    """
    src_data = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )
    src_mask = np.array(
        [
            [True, False],
            [False, False],
            [False, False],
        ]
    )
    src_array = ma.array(src_data, mask=src_mask)

    lon, lat, lon_bounds, lat_bounds = make_grid_args(2, 3)
    src_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    src_esmpy_grid = src_grid._make_esmf_sdo()
    src_esmpy_grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
    src_esmpy_grid.mask[0][...] = src_mask
    src_field = esmpy.Field(src_esmpy_grid)
    src_field.data[...] = src_data

    lon, lat, lon_bounds, lat_bounds = make_grid_args(3, 2)
    tgt_grid = GridInfo(lon, lat, lon_bounds, lat_bounds)
    tgt_field = tgt_grid.make_esmf_field()

    regridder = Regridder(src_grid, tgt_grid)

    regridding_kwargs = {
        "ignore_degenerate": True,
        "regrid_method": esmpy.RegridMethod.CONSERVE,
        "unmapped_action": esmpy.UnmappedAction.IGNORE,
        "factors": True,
        "src_mask_values": [1],
    }
    esmpy_fracarea_regridder = esmpy.Regrid(
        src_field, tgt_field, norm_type=esmpy.NormType.FRACAREA, **regridding_kwargs
    )
    esmpy_dstarea_regridder = esmpy.Regrid(
        src_field, tgt_field, norm_type=esmpy.NormType.DSTAREA, **regridding_kwargs
    )

    tgt_field_dstarea = esmpy_dstarea_regridder(src_field, tgt_field)
    result_esmpy_dstarea = tgt_field_dstarea.data
    result_dstarea = regridder.regrid(src_array, norm_type=Constants.NormType.DSTAREA)
    assert ma.allclose(result_esmpy_dstarea, result_dstarea)

    tgt_field_fracarea = esmpy_fracarea_regridder(src_field, tgt_field)
    result_esmpy_fracarea = tgt_field_fracarea.data
    result_fracarea = regridder.regrid(src_array, norm_type=Constants.NormType.FRACAREA)
    assert ma.allclose(result_esmpy_fracarea, result_fracarea)
