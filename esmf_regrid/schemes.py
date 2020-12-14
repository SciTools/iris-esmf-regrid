# -*- coding: utf-8 -*-
"""TBD: public module docstring."""

import iris

from ._grid import GridInfo
from .esmf_regridder import Regridder

__all__ = [
    "ESMF",
]


class ESMF:
    """TBD: public class docstring."""

    def __init__(self):
        pass

    def regridder(self, src_cube, tgt_cube):
        """TBD: public method docstring."""
        return _ESMFRegridder(src_cube, tgt_cube)


class _ESMFRegridder:
    def __init__(self, src_cube, tgt_cube, mdtol=1.):
        src_grid_info = GridInfo.from_cube(src_cube)
        tgt_grid_info = GridInfo.from_cube(tgt_cube)
        self.regridder = Regridder(src_grid_info, tgt_grid_info)
        self.mdtol = mdtol

    def __call__(self, src):
        data = self.regridder.regrid(src.core_data()[0], mdtol=self.mdtol)
        return iris.cube.Cube(data)
