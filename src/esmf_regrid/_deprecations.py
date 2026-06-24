"""Utilities for producing runtime deprecation messages."""

import warnings


def warn_deprecated(msg, stacklevel=2):
    """Issues a deprecation warning."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=stacklevel)
