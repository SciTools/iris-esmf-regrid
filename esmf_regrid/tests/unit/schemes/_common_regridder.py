"""Common conveniences for testing :mod:`esmf_regrid.schemes` regridder classes."""

from abc import ABC
from typing import Tuple

from cf_units import Unit
import numpy as np
import pytest

from esmf_regrid.tests.unit.schemes.test__cube_to_GridInfo import (
    _curvilinear_cube,
    _grid_cube,
)


class _CommonRegridder(ABC):
    """A class containing tests for a regridder, configurable via subclassing."""

    # Set REGRIDDER to the desired regridder class.
    REGRIDDER = NotImplemented
    # Configurable for the Nearest regridder.
    N_LONS_SRC = 6

    def test_dim_switching(self):
        """
        Test dimension order agnosticism.

        Checks that the regridder accepts a cube with dimensions in a different
        order than the cube which initialised it. Checks that dimension order is
        inherited from the cube in the calling function in both cases.
        """
        n_lons = 6
        n_lats = 5
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
        tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

        regridder = self.REGRIDDER(src, tgt)
        unswitched_result = regridder(src)

        src_switched = src.copy()
        src_switched.transpose()
        switched_result = regridder(src_switched)

        assert unswitched_result.coord(dimensions=(0,)).standard_name == "latitude"
        assert unswitched_result.coord(dimensions=(1,)).standard_name == "longitude"
        assert switched_result.coord(dimensions=(0,)).standard_name == "longitude"
        assert switched_result.coord(dimensions=(1,)).standard_name == "latitude"

    def test_differing_grids(self):
        """Test erroring when called on a different grid to the init grid."""
        n_lons = 6
        n_lats = 5
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        src_init = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
        tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

        n_lons_dif = 7
        src_dif_coord = _grid_cube(
            n_lons_dif, n_lats, lon_bounds, lat_bounds, circular=True
        )
        src_dif_circ = _grid_cube(
            n_lons, n_lats, lon_bounds, lat_bounds, circular=False
        )

        regridder = self.REGRIDDER(src_init, tgt)

        msg = "The given cube is not defined on the same source grid as this regridder."
        with pytest.raises(ValueError, match=msg):
            _ = regridder(src_dif_coord)
        with pytest.raises(ValueError, match=msg):
            _ = regridder(src_dif_circ)

    def test_invalid_mdtol(self):
        """Test erroring when mdtol is out of range."""
        n_lons = 6
        n_lats = 5
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        src = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)
        tgt = _grid_cube(n_lons, n_lats, lon_bounds, lat_bounds, circular=True)

        match = "Value for mdtol must be in range 0 - 1, got "
        with pytest.raises(ValueError, match=match):
            _ = self.REGRIDDER(src, tgt, mdtol=2)
        with pytest.raises(ValueError, match=match):
            _ = self.REGRIDDER(src, tgt, mdtol=-1)

    def test_curvilinear_equivalence(self):
        """Test curvilinear and rectilinear coordinates giving the same results."""
        n_lons_src = self.N_LONS_SRC
        n_lons_tgt = 3
        n_lats_src = 4
        n_lats_tgt = 2
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        grid_src = _grid_cube(
            n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True
        )
        grid_tgt = _grid_cube(
            n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True
        )
        curv_src = _curvilinear_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)
        curv_tgt = _curvilinear_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        grid_to_grid = self.REGRIDDER(grid_src, grid_tgt)
        grid_to_curv = self.REGRIDDER(grid_src, curv_tgt)
        curv_to_grid = self.REGRIDDER(curv_src, grid_tgt)
        curv_to_curv = self.REGRIDDER(curv_src, curv_tgt)

        def extract_weights(regridder):
            return regridder.regridder.weight_matrix.todense()

        for regridder in [grid_to_curv, curv_to_grid, curv_to_curv]:
            assert np.allclose(
                extract_weights(grid_to_grid), extract_weights(regridder)
            )

    def test_curvilinear_and_rectilinear(self):
        """
        Test handling of cubes with both curvilinear and rectilinear coords.

        Checks that the DimCoords have priority over AuxCoords.
        """
        n_lons_src = self.N_LONS_SRC
        n_lons_tgt = 3
        n_lats_src = 4
        n_lats_tgt = 2
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        grid_src = _grid_cube(
            n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True
        )
        grid_tgt = _grid_cube(
            n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True
        )
        curv_src = _curvilinear_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)
        curv_tgt = _curvilinear_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)

        src = curv_src.copy()
        grid_lat_src = grid_src.coord("latitude")
        grid_lat_src.standard_name = "grid_latitude"
        src.add_dim_coord(grid_lat_src, 0)
        grid_lon_src = grid_src.coord("longitude")
        grid_lon_src.standard_name = "grid_longitude"
        src.add_dim_coord(grid_lon_src, 1)

        tgt = curv_tgt.copy()
        grid_lat_tgt = grid_tgt.coord("latitude")
        grid_lat_tgt.standard_name = "grid_latitude"
        tgt.add_dim_coord(grid_lat_tgt, 0)
        grid_lon_tgt = grid_tgt.coord("longitude")
        grid_lon_tgt.standard_name = "grid_longitude"
        tgt.add_dim_coord(grid_lon_tgt, 1)

        # Change the AuxCoords to check that the DimCoords have priority.
        src.coord("latitude").bounds[:] = 0
        src.coord("longitude").bounds[:] = 0
        tgt.coord("latitude").bounds[:] = 0
        tgt.coord("longitude").bounds[:] = 0

        # Ensure the source data is all ones.
        src.data[:] = 1

        rg = self.REGRIDDER(src, tgt)
        result = rg(src)

        expected = grid_tgt.copy()
        # If the aux coords had been prioritised, expected.data would be a fully masked array.
        expected.data[:] = 1
        assert expected == result
        assert not np.ma.is_masked(result)

    def test_unit_equivalence(self):
        """Test that equivalent coordinates in degrees/radians give same results."""
        n_lons_src = self.N_LONS_SRC
        n_lons_tgt = 3
        n_lats_src = 4
        n_lats_tgt = 2
        lon_bounds = (-180, 180)
        lat_bounds = (-90, 90)
        lon_rad_bounds = (-np.pi, np.pi)
        lat_rad_bounds = (-np.pi / 2, np.pi / 2)

        def rad_coords(cube):
            cube.coord("latitude").units = Unit("radians")
            cube.coord("longitude").units = Unit("radians")

        grid_src = _grid_cube(
            n_lons_src, n_lats_src, lon_bounds, lat_bounds, circular=True
        )
        grid_src_rad = _grid_cube(
            n_lons_src, n_lats_src, lon_rad_bounds, lat_rad_bounds, circular=True
        )
        rad_coords(grid_src_rad)
        grid_tgt = _grid_cube(
            n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds, circular=True
        )
        grid_tgt_rad = _grid_cube(
            n_lons_tgt, n_lats_tgt, lon_rad_bounds, lat_rad_bounds, circular=True
        )
        rad_coords(grid_tgt_rad)
        curv_src = _curvilinear_cube(n_lons_src, n_lats_src, lon_bounds, lat_bounds)
        curv_src_rad = _curvilinear_cube(
            n_lons_src, n_lats_src, lon_rad_bounds, lat_rad_bounds
        )
        rad_coords(curv_src_rad)
        curv_tgt = _curvilinear_cube(n_lons_tgt, n_lats_tgt, lon_bounds, lat_bounds)
        curv_tgt_rad = _curvilinear_cube(
            n_lons_tgt, n_lats_tgt, lon_rad_bounds, lat_rad_bounds
        )
        rad_coords(curv_tgt_rad)

        grid_to_grid = self.REGRIDDER(grid_src, grid_tgt)
        grid_rad_to_grid = self.REGRIDDER(grid_src_rad, grid_tgt)
        grid_rad_to_curv = self.REGRIDDER(grid_src_rad, curv_tgt)
        curv_to_grid_rad = self.REGRIDDER(curv_src, grid_tgt_rad)
        curv_rad_to_grid = self.REGRIDDER(curv_src_rad, grid_tgt)
        curv_to_curv_rad = self.REGRIDDER(curv_src, curv_tgt_rad)

        def extract_weights(regridder):
            return regridder.regridder.weight_matrix.todense()

        for regridder in [
            grid_rad_to_grid,
            grid_rad_to_curv,
            curv_to_grid_rad,
            curv_rad_to_grid,
            curv_to_curv_rad,
        ]:
            assert np.allclose(
                extract_weights(grid_to_grid), extract_weights(regridder)
            )

    def _test_masks(
        self, src_indexing: Tuple[slice, ...], tgt_indexing: Tuple[slice, ...]
    ):
        """
        Check that the `use_src_mask` and `use_tgt_mask` keywords work properly.

        Designed for subclass tests to call this method.
        """
        src = _curvilinear_cube(7, 6, [-180, 180], [-90, 90])
        tgt = _curvilinear_cube(6, 7, [-180, 180], [-90, 90])

        # Make src and tgt discontiguous at (0, 0)
        src_mask = np.zeros([6, 7], dtype=bool)
        src_mask[0, 0] = True
        src.data = np.ma.array(src.data, mask=src_mask)
        src_discontiguous = src.copy()
        src_discontiguous.coord("latitude").bounds[0, 0] = 0
        src_discontiguous.coord("longitude").bounds[0, 0] = 0

        tgt_mask = np.zeros([7, 6], dtype=bool)
        tgt_mask[0, 0] = True
        tgt.data = np.ma.array(tgt.data, mask=tgt_mask)
        tgt_discontiguous = tgt.copy()
        tgt_discontiguous.coord("latitude").bounds[0, 0] = 0
        tgt_discontiguous.coord("longitude").bounds[0, 0] = 0

        rg_src_masked = self.REGRIDDER(src_discontiguous, tgt, use_src_mask=True)
        rg_tgt_masked = self.REGRIDDER(src, tgt_discontiguous, use_tgt_mask=True)
        rg_unmasked = self.REGRIDDER(src, tgt)

        weights_src_masked = rg_src_masked.regridder.weight_matrix
        weights_tgt_masked = rg_tgt_masked.regridder.weight_matrix
        weights_unmasked = rg_unmasked.regridder.weight_matrix

        # Check there are no weights associated with the masked point.
        assert weights_src_masked[:, 0].nnz == 0
        assert weights_tgt_masked[0].nnz == 0

        # Check all other weights are correct.
        np.testing.assert_allclose(
            weights_src_masked[src_indexing].todense(),
            weights_unmasked[src_indexing].todense(),
        )
        np.testing.assert_allclose(
            weights_tgt_masked[tgt_indexing].todense(),
            weights_unmasked[tgt_indexing].todense(),
        )
