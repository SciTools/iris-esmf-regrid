"""Provides ESMF representations of UGRID meshes."""

import ESMF
import numpy as np

from .._esmf_sdo import SDO


class MeshInfo(SDO):
    """
    Class for handling unstructured meshes.

    This class holds information about Meshes in a form similar to UGRID.
    It contains methods for translating this information into ESMF objects.
    In particular, there are methods for representing as an ESMF Mesh and
    as an ESMF Field containing that Mesh. This ESMF Field is designed to
    contain enough information for area weighted regridding and may be
    inappropriate for other ESMF regridding schemes.
    """

    def __init__(
        self,
        node_coords,
        face_node_connectivity,
        node_start_index,
        elem_start_index=0,
        areas=None,
    ):
        """
        Create a MeshInfo object describing a UGRID-like mesh.

        Parameters
        ----------
        node_coords: array_like
            An Nx2 numpy array describing the location of the nodes of the mesh.
            node_coords[:,0] describes the longitudes in degrees and
            node_coords[:,1] describes the latitudes in degrees
        face_node_connectivity: array_like
            A numpy masked array describing the face node connectivity of the
            mesh. The unmasked points of face_node_connectivity[i] describe
            which nodes are connected to the i'th face.
        node_start_index: int
            An integer the value which, appearing in the face_node_connectivity
            array, indicates the first node in the node_coords array.
            UGRID supports both 0 based and 1 based indexing, so both must be
            accounted for here:
            https://ugrid-conventions.github.io/ugrid-conventions/#zero-or-one
        elem_start_index: int, optional
            An integer describing what index should be considered by ESMF to be
            the start index for describing its elements. This makes no
            difference to the regridding calculation and will only affect the
            intermediate ESMF objects, should the user need access to them.
            Defaults to 0.
        areas: array_like, optional
            Either None or a numpy array describing the areas associated with
            each face. If None, then ESMF will use its own calculated areas.
            Defaults to None.
        """
        self.node_coords = node_coords
        self.fnc = face_node_connectivity
        self.nsi = node_start_index
        self.esi = elem_start_index
        self.areas = areas
        super().__init__(
            shape=(len(face_node_connectivity),),
            index_offset=self.esi,
            field_kwargs={"meshloc": ESMF.MeshLoc.ELEMENT},
        )

    def _as_esmf_info(self):
        # ESMF uses a slightly different format to UGRID,
        # the data must be translated into a form ESMF understands
        num_node = self.node_coords.shape[0]
        num_elem = self.fnc.shape[0]
        nodeId = np.array(range(self.nsi, self.nsi + num_node))
        nodeCoord = self.node_coords.flatten()
        nodeOwner = np.zeros([num_node])  # regridding currently serial
        elemId = np.array(range(self.esi, self.esi + num_elem))
        elemType = self.fnc.count(axis=1)
        # Experiments seem to indicate that ESMF is using 0 indexing here
        elemConn = self.fnc.compressed() - self.nsi
        result = (
            num_node,
            num_elem,
            nodeId,
            nodeCoord,
            nodeOwner,
            elemId,
            elemType,
            elemConn,
            self.areas,
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
            areas,
        ) = info
        # ESMF can handle other dimensionalities, but we are unlikely
        # to make such a use of ESMF
        emesh = ESMF.Mesh(
            parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_DEG
        )

        emesh.add_nodes(num_node, nodeId, nodeCoord, nodeOwner)
        emesh.add_elements(num_elem, elemId, elemType, elemConn, element_area=areas)
        return emesh
