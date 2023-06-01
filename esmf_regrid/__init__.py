try:
    import esmpy
except ImportError as exc:
    # Prior to v8.4.0, `esmpy`` could be imported as `ESMF`.
    try:
        import ESMF as esmpy  # noqa: N811
    except ImportError:
        raise exc

# constants is used within schemes, so needs to imported first
from ._constants import Constants
from .schemes import *


__version__ = "0.6.0"
