"""Unit tests for :class:`esmf_regrid.schemes.ESMFAreaWeighted`."""

from esmf_regrid.schemes import ESMFAreaWeighted
from ._common_scheme import _CommonScheme


class TestAreaWeighted(_CommonScheme):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFAreaWeighted`."""

    METHOD = ESMFAreaWeighted
