"""Holds all enums created for esmf-regrid."""

from enum import Enum

from . import esmpy


class Constants:
    """Encompassing class for best practice import."""

    class Method(Enum):
        """Holds enums for Method values."""

        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    class NormType(Enum):
        """Holds enums for norm types."""

        FRACAREA = esmpy.api.constants.NormType.FRACAREA
        DSTAREA = esmpy.api.constants.NormType.DSTAREA


method_dict = {
    "conservative": Constants.Method.CONSERVATIVE,
    "bilinear": Constants.Method.BILINEAR,
    "nearest": Constants.Method.NEAREST,
}

norm_dict = {
    "fracarea": Constants.NormType.FRACAREA,
    "dstarea": Constants.NormType.DSTAREA,
}


def check_method(method):
    """Check that method is a member of the `Constants.Method` enum or raise an error."""
    if method in method_dict:
        result = method_dict[method]
    elif method in method_dict.values():
        result = method
    else:
        e_msg = (
            f"Method must be a member of `Constants.Method` enum, instead got {method}"
        )
        raise ValueError(e_msg)
    return result


def check_norm(norm):
    """Check that normtype is a member of the `Constants.NormType` enum or raise an error."""
    if norm in norm_dict:
        result = norm_dict[norm]
    elif norm in norm_dict.values():
        result = norm
    else:
        e_msg = f"NormType must be a member of `Constants.NormType` enum, instead got {norm}"
        raise ValueError(e_msg)
    return result
