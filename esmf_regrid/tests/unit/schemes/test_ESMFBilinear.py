"""Unit tests for :class:`esmf_regrid.schemes.ESMFBilinear`."""

from esmf_regrid.schemes import ESMFBilinear
from ._common_scheme import _CommonScheme


class TestBilinear(_CommonScheme):
    """Run the common scheme tests against :class:`esmf_regrid.schemes.ESMFBilinear`."""

    SCHEME = ESMFBilinear
