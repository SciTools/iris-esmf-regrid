# Copyright Iris-esmf-regrid contributors
#
# This file is part of Iris-esmf-regrid and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for producing runtime deprecation messages."""

import warnings


def warn_deprecated(msg, stacklevel=2):
    """Issue a deprecation warning."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=stacklevel)
