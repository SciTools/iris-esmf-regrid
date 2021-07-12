"""Unit tests for :mod:`esmf_regrid.experimental.unstructured_scheme`."""

try:
    import iris.experimental.ugrid

    iris.experimental.ugrid.Mesh
except:
    msg = "skipping tests which use unstructured iris cubes"
    pytestmark = pytest.mark.skip(msg)
