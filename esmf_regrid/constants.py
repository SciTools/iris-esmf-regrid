"""Holds all enums created for esmf-regrid."""

from enum import Enum

from . import esmpy

class Constants:
    """Encompassing class for best practice import."""
    class Method(Enum):
        """holds enums for Method values."""
        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    NormType = Enum("NormType", ["FRACAREA", "DSTAREA"])
