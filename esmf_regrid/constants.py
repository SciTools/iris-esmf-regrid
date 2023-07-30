from enum import Enum
from . import esmpy

class Constants:
    class Method(Enum):
        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    NormType = Enum("NormType", ["FRACAREA", "DSTAREA"])