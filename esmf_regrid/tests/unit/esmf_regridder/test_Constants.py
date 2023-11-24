"""Unit tests for :class:`esmf_regrid.constants.Constants`, and associated checkers."""

import pytest

from esmf_regrid import check_method, check_norm, Constants


@pytest.mark.parametrize(
    "method",
    [
        ["conservative", Constants.Method.CONSERVATIVE],
        ["bilinear", Constants.Method.BILINEAR],
        ["nearest", Constants.Method.NEAREST],
    ],
)
def test_check_method_validates(method):
    """Tests that `check_method()` converts strings to enum."""
    # with original text-input behaviour
    assert check_method(method[0]) == method[1]
    # with updated enum-input behaviour
    assert check_method(method[1]) == method[1]


def test_invalid_method():
    """Ensures invalid methods raise an appropriate error, via string and enum."""
    with pytest.raises(ValueError):
        _ = check_method("other")
    with pytest.raises(AttributeError):
        _ = check_method(Constants.Method.OTHER)


@pytest.mark.parametrize(
    "norm",
    [
        ["fracarea", Constants.NormType.FRACAREA],
        ["dstarea", Constants.NormType.DSTAREA],
    ],
)
def test_check_norm_validates(norm):
    """Tests that `check_norm()` converts strings to enum."""
    # with original text-input behaviour
    assert check_norm(norm[0]) == norm[1]
    # with updated enum-input behaviour
    assert check_norm(norm[1]) == norm[1]


def test_invalid_norm():
    """Ensures invalid norms raise an appropriate error, via string and enum."""
    with pytest.raises(ValueError):
        _ = check_norm("other")
    with pytest.raises(AttributeError):
        _ = check_norm(Constants.Norm.OTHER)
