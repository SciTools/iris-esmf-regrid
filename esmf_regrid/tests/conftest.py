"""Common testing infrastructure."""

import pytest


@pytest.fixture(params=["float32", "float64"])
def in_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        ("grid", "grid"),
        ("grid", "mesh"),
        ("mesh", "grid"),
        ("mesh", "mesh"),
    ]
)
def src_tgt_types(request):
    return request.param
