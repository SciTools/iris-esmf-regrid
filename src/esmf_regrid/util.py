"""Miscellaneous utility functions."""

from esmf_regrid.schemes import _cube_to_GridInfo, _mesh_to_MeshInfo


def find_area(cube, radius=1):
    """Return the areas of cells on a cube.

    Defaults to the area of each cell on a unit sphere, but actual
    radius can be specified.

    Parameters
    ----------
    cube : cube
        Cube containing the mesh or grid to calculate the area of.
    radius : float, default=1
        Radius of the sphere used to calculate area.
    """
    if cube.mesh is not None:
        assert cube.location == "face"
        sdo = _mesh_to_MeshInfo(cube.mesh, cube.location)
    else:
        sdo = _cube_to_GridInfo(cube)
    field = sdo.make_esmf_field()
    field.get_area()
    areas = field.data.copy()
    field.destroy()
    areas *= radius
    return areas
