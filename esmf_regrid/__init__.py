try:
    import esmpy
except ImportError as exc:
    # Prior to v8.4.0, `esmpy`` could be imported as `ESMF`.
    try:
        import ESMF as esmpy  # noqa: N811
    except ImportError:
        raise exc

from .schemes import *
from .constants import Constants

__version__ = "0.6.0"
