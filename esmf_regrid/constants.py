from enum import Enum, EnumMeta
from . import esmpy


class Constants:
    # used to ovveride AtrributeError into NotImplementedError
    class MethodMeta(EnumMeta):
        def __getattr__(self, name):
            try:
                super().__getattribute__(name)
            except AttributeError:
                raise NotImplementedError(
                    "The method you have chosen hasn't been implemented yet. "
                    "Must be a member of the Method enum."
                )

    class Method(Enum, metaclass=MethodMeta):
        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    NormType = Enum("NormType", ["FRACAREA", "DSTAREA"])
