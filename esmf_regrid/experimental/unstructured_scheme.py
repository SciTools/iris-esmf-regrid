"""Provides an iris interface for unstructured regridding."""

import iris
from iris.analysis._interpolation import get_xy_coords
import numpy as np

# from numpy import ma

from esmf_regrid.esmf_regridder import GridInfo, Regridder

# from esmf_regrid.experimental.unstructured_regrid import MeshInfo

# Taken from PR #26
def _bounds_cf_to_simple_1d(cf_bounds):
    assert (cf_bounds[1:, 0] == cf_bounds[:-1, 1]).all()
    simple_bounds = np.empty((cf_bounds.shape[0] + 1,), dtype=np.float64)
    simple_bounds[:-1] = cf_bounds[:, 0]
    simple_bounds[-1] = cf_bounds[-1, 1]
    return simple_bounds


def _get_mesh_and_dim(cube):
    # Returns the cube's mesh and the dimension that mesh belongs to.
    # Likely to be of the form:
    # mesh = cube.mesh
    # mesh_dim = cube.mesh_dim(mesh)
    # return mesh, mesh_dim
    pass


def _cube_to_MeshInfo(cube):
    # Returns a MeshInfo object describing the mesh of the cube.
    pass


def _cube_to_GridInfo(cube):
    # Returns a GridInfo object describing the horizontal grid of the cube.
    # This may be inherited from code written for the rectilinear regridding scheme.
    lat = cube.coord("latitude")
    lon = cube.coord("longitude")
    # TODO: accomodate other x/y coords.
    # TODO: perform checks on lat/lon.
    return GridInfo(
        lon.points,
        lat.points,
        _bounds_cf_to_simple_1d(lon.bounds),
        _bounds_cf_to_simple_1d(lat.bounds),
        circular=lon.circular,
    )


# def _regrid_along_dims(regridder, data, src_dim, mdtol):
#     # Before regridding, data is transposed to a standard form.
#     # This will be done either with something like the following code
#     # or else done within the regridder by specifying args.
#     # new_axes = list(range(len(data.shape)))
#     # new_axes.pop(src_dim)
#     # new_axes.append(src_dim)
#     # data = ma.transpose(data, axes=new_axes)
#
#     result = regridder.regrid(data, mdtol=mdtol)
#     return result


def _create_cube(data, src_cube, mesh_dim, mesh, grid_x, grid_y):
    # Here we expect the args to be as follows:
    # data: a masked array containing the result of the regridding operation
    # src_cube: the source cube which data is regrid from
    # mesh_dim: the dimension on src_cube which the mesh belongs to
    # mesh: the Mesh (or MeshCoord) object belonging to src_cube
    # grid_x: the coordinate on the target cube representing the x axis
    # grid_y: the coordinate on the target cube representing the y axis

    new_cube = iris.cube.Cube(data)

    # TODO: add coords and metadata.

    return new_cube


def _regrid_unstructured_to_rectilinear__prepare(src_mesh_cube, target_grid_cube):
    # TODO: Perform checks on the arguments. (grid coords are contiguous,
    #  spherical and monotonic. Mesh is defined on faces)

    # TODO: Account for differences in units.

    # TODO: Account for differences in coord systems.

    # TODO: Record appropriate dimensions (i.e. which dimension the mesh belongs to)

    grid_x, grid_y = get_xy_coords(target_grid_cube)

    meshinfo = _cube_to_MeshInfo(src_mesh_cube)
    gridinfo = _cube_to_GridInfo(target_grid_cube)

    regridder = Regridder(meshinfo, gridinfo)

    mesh, mesh_dim = _get_mesh_and_dim(src_mesh_cube)

    regrid_info = (mesh, mesh_dim, grid_x, grid_y, regridder)

    return regrid_info


def _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol):
    mesh, mesh_dim, grid_x, grid_y, regridder = regrid_info

    # Perform regridding with realised data for the moment. This may be changed
    # in future to handle src_cube.lazy_data.
    new_data = regridder.regrid(src_cube.data, mdtol=mdtol)
    # When we want to handle extra dimensions, we may want to do something like:
    # new_data = _regrid_along_dims(src_cube.data, mesh_dim, mdtol)

    new_cube = _create_cube(
        new_data,
        src_cube,
        mesh_dim,
        mesh,
        grid_x,
        grid_y,
    )

    # TODO: apply tweaks to created cube (slice out length 1 dimensions)

    return new_cube


def regrid_unstructured_to_rectilinear(src_cube, grid_cube, mdtol=0):
    """TODO: write docstring."""
    regrid_info = _regrid_unstructured_to_rectilinear__prepare(src_cube, grid_cube)
    result = _regrid_unstructured_to_rectilinear__perform(src_cube, regrid_info, mdtol)
    return result


class MeshToGridESMFRegridder:
    """TODO: write docstring."""

    def __init__(self, src_mesh_cube, target_grid_cube, mdtol=1):
        """TODO: write docstring."""
        # TODO: Record information about the identity of the mesh. This would
        #  typically be a copy of the mesh, though given the potential size of
        #  the mesh, it may make sense to either retain a reference to the actual
        #  mesh or else something like a hash of the mesh.

        # Missing data tolerance.
        # Code directly copied from iris.
        if not (0 <= mdtol <= 1):
            msg = "Value for mdtol must be in range 0 - 1, got {}."
            raise ValueError(msg.format(mdtol))
        self.mdtol = mdtol

        partial_regrid_info = _regrid_unstructured_to_rectilinear__prepare(
            src_mesh_cube, target_grid_cube
        )

        # Store regrid info.
        _, _, self.grid_x, self.grid_y, self.regridder = partial_regrid_info

    def __call__(self, cube):
        """TODO: write docstring."""
        # TODO: Ensure cube has the same mesh as that of the recorded mesh.

        # mesh is probably an iris Mesh object, though it could also be a MeshCoord
        mesh, mesh_dim = _get_mesh_and_dim(cube)

        regrid_info = (mesh, mesh_dim, self.grid_x, self.grid_y, self.regridder)

        return _regrid_unstructured_to_rectilinear__perform(
            cube, regrid_info, self.mdtol
        )
