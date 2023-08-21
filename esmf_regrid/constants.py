from enum import Enum, EnumMeta
from . import esmpy


class Constants:
    class Method(Enum):  # metaclass=MethodMeta)
        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    NormType = Enum("NormType", ["FRACAREA", "DSTAREA"])
