"""Unit tests for :mod:`esmf_regrid.experimental.io.save_regridder`."""

import pytest

from esmf_regrid.experimental.io import save_regridder, _managed_var_name
from esmf_regrid.schemes import ESMFAreaWeightedRegridder
from esmf_regrid.tests.unit.schemes.test__mesh_to_MeshInfo import (
    _gridlike_mesh_cube,
)


def test_invalid_type(tmp_path):
    """Test that `save_regridder` raises a TypeError where appropriate."""
    invalid_obj = None
    filename = tmp_path / "regridder.nc"
    with pytest.raises(TypeError):
        save_regridder(invalid_obj, filename)


def test_var_name_preserve(tmp_path):
    """Test that `save_regridder` does not change var_ames."""
    lons = 3
    lats = 4
    src = _gridlike_mesh_cube(lons, lats)
    tgt = _gridlike_mesh_cube(lons, lats)

    DUMMY_VAR_NAME_SRC = "src_dummy_var"
    DUMMY_VAR_NAME_TGT = "tgt_dummy_var"
    for coord in src.mesh.coords():
        coord.var_name = DUMMY_VAR_NAME_SRC
    for coord in tgt.mesh.coords():
        coord.var_name = DUMMY_VAR_NAME_TGT

    rg = ESMFAreaWeightedRegridder(src, tgt)
    filename = tmp_path / "regridder.nc"
    save_regridder(rg, filename)

    for coord in src.mesh.coords():
        assert coord.var_name == DUMMY_VAR_NAME_SRC
    for coord in tgt.mesh.coords():
        assert coord.var_name == DUMMY_VAR_NAME_TGT


def test_managed_var_name():
    """Test that `_managed_var_name` changes var_names."""
    lons = 3
    lats = 4
    src = _gridlike_mesh_cube(lons, lats)
    tgt = _gridlike_mesh_cube(lons, lats)

    DUMMY_VAR_NAME_SRC = "src_dummy_var"
    DUMMY_VAR_NAME_TGT = "tgt_dummy_var"
    for coord in src.mesh.coords():
        coord.var_name = DUMMY_VAR_NAME_SRC
    for coord in tgt.mesh.coords():
        coord.var_name = DUMMY_VAR_NAME_TGT

    with _managed_var_name(src, tgt):
        for coord in src.mesh.coords():
            assert coord.var_name != DUMMY_VAR_NAME_SRC
        for coord in tgt.mesh.coords():
            assert coord.var_name != DUMMY_VAR_NAME_TGT
