"""Provides :mod:`esmpy` representations of UGRID meshes."""

import numpy as np

from .. import esmpy
from .._esmf_sdo import SDO


class MeshInfo(SDO):
    """Class for handling unstructured meshes.

    This class holds information about Meshes in a form similar to UGRID.
    It contains methods for translating this information into :mod:`esmpy` objects.
    In particular, there are methods for representing as an :class:`esmpy.api.mesh.Mesh`
    and as an :class:`esmpy.api.field.Field` containing that
    :class:`~esmpy.api.mesh.Mesh`. This esmpy :class:`~esmpy.api.field.Field` is designed to
    contain enough information for area weighted regridding and may be
    inappropriate for other :mod:`esmpy` regridding schemes.
    """

    def __init__(
        self,
        node_coords,
        face_node_connectivity,
        node_start_index,
        elem_start_index=0,
        areas=None,
        mask=None,
        elem_coords=None,
        location="face",
    ):
        """Create a :class:`MeshInfo` object describing a UGRID-like mesh.

        Parameters
        ----------
        node_coords: :obj:`~numpy.typing.ArrayLike`
            An ``Nx2`` array describing the location of the nodes of the mesh.
            ``node_coords[:,0]`` describes the longitudes in degrees and
            ``node_coords[:,1]`` describes the latitudes in degrees.
        face_node_connectivity: :obj:`~numpy.typing.ArrayLike`
            A masked array describing the face node connectivity of the
            mesh. The unmasked points of ``face_node_connectivity[i]`` describe
            which nodes are connected to the ``i``'th face.
        node_start_index: int
            A value which, appearing in the ``face_node_connectivity``
            array, indicates the first node in the ``node_coords`` array.
            UGRID supports both ``0`` based and ``1`` based indexing, so both must be
            accounted for here:
            https://ugrid-conventions.github.io/ugrid-conventions/#zero-or-one
        elem_start_index: int, default=0
            Describes what index should be considered by :mod:`esmpy` to be
            the start index for describing its elements. This makes no
            difference to the regridding calculation and will only affect the
            intermediate :mod:`esmpy` objects, should the user need access to them.
        areas: :obj:`~numpy.typing.ArrayLike`, optional
            Array describing the areas associated with
            each face. If ``None``, then :mod:`esmpy` will use its own calculated areas.
        mask: :obj:`~numpy.typing.ArrayLike`, optional
            Array describing which elements :mod:`esmpy` will ignore.
        elem_coords : :obj:`~numpy.typing.ArrayLike`, optional
            An ``Nx2`` array describing the location of the face centers of the mesh.
            ``elem_coords[:,0]`` describes the longitudes in degrees and
            ``elem_coords[:,1]`` describes the latitudes in degrees.
        location : str, default="face"
            Either "face" or "node". Describes the location for data on the mesh.
        """
        self.node_coords = node_coords
        self.fnc = face_node_connectivity
        self.nsi = node_start_index
        self.esi = elem_start_index
        self.areas = areas
        self.elem_coords = elem_coords
        if location == "face":
            field_kwargs = {"meshloc": esmpy.MeshLoc.ELEMENT}
            shape = (len(face_node_connectivity),)
        elif location == "node":
            field_kwargs = {"meshloc": esmpy.MeshLoc.NODE}
            shape = (len(node_coords),)
        else:
            raise ValueError(
                f"The mesh location '{location}' is not supported, only "
                f"'face' and 'node' are supported."
            )
        super().__init__(
            shape=shape,
            index_offset=1,
            field_kwargs=field_kwargs,
            mask=mask,
        )

    def _as_esmf_info(self):
        # ESMF uses a slightly different format to UGRID,
        # the data must be translated into a form ESMF understands
        num_node = self.node_coords.shape[0]
        num_elem = self.fnc.shape[0]
        nodeId = np.arange(1, num_node + 1)
        nodeCoord = self.node_coords.flatten()
        nodeOwner = np.zeros([num_node])  # regridding currently serial
        elemId = np.arange(1, num_elem + 1)
        if np.ma.isMaskedArray(self.fnc):
            elemType = self.fnc.count(axis=1)
            # Experiments seem to indicate that ESMF is using 0 indexing here
            elemConn = self.fnc.compressed() - self.nsi
        else:
            elemType = np.full(self.fnc.shape[:1], self.fnc.shape[1])
            # Experiments seem to indicate that ESMF is using 0 indexing here
            elemConn = self.fnc.flatten() - self.nsi
        elemCoord = self.elem_coords
        result = (
            num_node,
            num_elem,
            nodeId,
            nodeCoord,
            nodeOwner,
            elemId,
            elemType,
            elemConn,
            self.mask,
            self.areas,
            elemCoord,
        )
        return result

    def _make_esmf_sdo(self):
        info = self._as_esmf_info()
        (
            num_node,
            num_elem,
            nodeId,
            nodeCoord,
            nodeOwner,
            elemId,
            elemType,
            elemConn,
            mask,
            areas,
            elemCoord,
        ) = info
        # ESMF can handle other dimensionalities, but we are unlikely
        # to make such a use of ESMF
        emesh = esmpy.Mesh(
            parametric_dim=2, spatial_dim=2, coord_sys=esmpy.CoordSys.SPH_DEG
        )

        emesh.add_nodes(num_node, nodeId, nodeCoord, nodeOwner)
        emesh.add_elements(
            num_elem,
            elemId,
            elemType,
            elemConn,
            element_mask=mask,
            element_area=areas,
            element_coords=elemCoord,
        )
        return emesh
