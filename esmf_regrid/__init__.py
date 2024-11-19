try:
    import esmpy
except ImportError as exc:
    # Prior to v8.4.0, `esmpy`` could be imported as `ESMF`.
    try:
        import ESMF as esmpy  # noqa: N811
    except ImportError:
        raise exc

try:
    import iris.mesh as _imesh
except ImportError as exc:
    try:
        import iris.experimental.ugrid as _imesh
    except ImportError:
        raise exc

if hasattr(_imesh, "PARSE_UGRID_ON_LOAD"):
    _load_context = _imesh.PARSE_UGRID_ON_LOAD.context
else:
    from contextlib import nullcontext

    _load_context = nullcontext

# constants needs to be above schemes, as it is used within
from .constants import Constants, check_method, check_norm
from .schemes import *

__version__ = "0.12.dev0"
