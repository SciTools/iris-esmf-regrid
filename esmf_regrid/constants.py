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

    class NormType(Enum):
        """holds enums for norm types."""
        FRACAREA = esmpy.api.constants.NormType.FRACAREA
        DSTAREA = esmpy.api.constants.NormType.DSTAREA

method_dict = {
    "conservative": Constants.Method.CONSERVATIVE,
    "bilinear": Constants.Method.BILINEAR,
    "nearest": Constants.Method.NEAREST}

norm_dict = {
    "fracarea": Constants.NormType.FRACAREA,
    "dstarea": Constants.NormType.DSTAREA}

def check_method(method):
    if method in method_dict.keys():
        result = method_dict[method]
    elif method in method_dict.values():
        result = method
    else:
        raise ValueError(f"Method must be a member of `Constants.Method` enum, instead got {method}")
    return result

def check_norm(norm):
    if norm in norm_dict.keys():
        result = norm_dict[norm]
    elif norm in norm_dict.values():
        result = norm
    else:
        raise ValueError(f"NormType must be a member of `Constants.NormType` enum, instead got {norm}")
    return result
