from enum import Enum
from . import esmpy


class Constants:
    class Method(Enum):
        CONSERVATIVE = esmpy.RegridMethod.CONSERVE
        BILINEAR = esmpy.RegridMethod.BILINEAR
        NEAREST = esmpy.RegridMethod.NEAREST_STOD

    NormType = Enum("NormType", ["FRACAREA", "DSTAREA"])
    # used in other files, placed here to have them all in one place
    Location = Enum("Location", ["FACE", "NODE"])
