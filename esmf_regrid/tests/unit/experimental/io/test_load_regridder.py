"""Unit tests for :mod:`esmf_regrid.experimental.io.save_regridder`."""

from esmf_regrid.experimental.io import load_regridder
from esmf_regrid.experimental.unstructured_scheme import GridToMeshESMFRegridder
import numpy as np
from numpy import ma

from .test_round_tripping import (
    _compare_ignoring_var_names,
    _make_grid_to_mesh_regridder,
)


def test_load_v09(request):
    """Test the loading of a file saved in version 0.9.0."""
    filename = request.path.parent / "v0.9_regridder.nc"
    method = "conservative"
    regridder = GridToMeshESMFRegridder
    equivalent_rg, src = _make_grid_to_mesh_regridder(
        method=method, regridder=regridder, circular=True
    )
    loaded_rg = load_regridder(str(filename))

    assert equivalent_rg.location == loaded_rg.location
    assert equivalent_rg.method == loaded_rg.method
    assert equivalent_rg.mdtol == loaded_rg.mdtol
    _compare_ignoring_var_names(equivalent_rg.grid_x, loaded_rg.grid_x)
    _compare_ignoring_var_names(equivalent_rg.grid_y, loaded_rg.grid_y)

    # Compare the weight matrices.
    original_matrix = equivalent_rg.regridder.weight_matrix
    loaded_matrix = loaded_rg.regridder.weight_matrix
    # Ensure the original and loaded weight matrix have identical type.
    assert type(original_matrix) is type(loaded_matrix)  # E721
    assert np.array_equal(original_matrix.todense(), loaded_matrix.todense())

    # Demonstrate regridding still gives the same results.
    src_data = ma.arange(np.prod(src.data.shape)).reshape(src.data.shape)
    src_data[0, 0] = ma.masked
    src.data = src_data
    # TODO: make this a cube comparison when mesh comparison becomes available.
    original_result = equivalent_rg(src).data
    loaded_result = loaded_rg(src).data
    assert np.array_equal(original_result, loaded_result)
    assert np.array_equal(original_result.mask, loaded_result.mask)

    # Ensure versions match what was saved.
    expected_esmf_version = "8.4.2"
    expected_save_version = "0.9.0"
    assert expected_esmf_version == loaded_rg.regridder.esmf_version
    assert expected_save_version == loaded_rg.regridder.esmf_regrid_version
