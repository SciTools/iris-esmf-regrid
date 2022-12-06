import numpy as np
from numpy import ma
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.experimental.ugrid import Connectivity, Mesh


def node_to_directed_edge(e2n):
    jagged_n2de = [[] for _ in range(e2n.max() + 1)]
    for edge_index, node_index in enumerate(e2n[:, ::-1].T.flatten()):
        jagged_n2de[node_index].append(edge_index)
    return jagged_n2de


def node_from_directed_edge(e2n):
    jagged_n2de = [[] for _ in range(e2n.max() + 1)]
    for edge_index, node_index in enumerate(e2n.T.flatten()):
        jagged_n2de[node_index].append(edge_index)
    return jagged_n2de


def face_to_deirected_edge(f2n, e2n):
    ntde = node_to_directed_edge(e2n)
    nfde = node_from_directed_edge(e2n)
    ef2n = ma.concatenate([f2n, f2n[:, :1]], axis=1)
    f2de = []
    for face in ef2n:
        edges = []
        face = face.compressed()
        for start, end in zip(face[:-1], face[1:]):
            out_edges = nfde[start]
            in_edges = ntde[end]
            edge = (set(out_edges) & set(in_edges)).pop()
            edges.append(edge)
        f2de.append(edges)
    return f2de


def face_to_edge(f2de, ne):
    return [[de % ne for de in run] for run in f2de]


def node_to_ac_edges(n2de, f2de, ne):
    de2de = ma.array(np.zeros(2 * ne, dtype=int), mask=True)
    for des in f2de:
        for i in range(len(des)):
            de2de[des[i - 1]] = (des[i] + ne) % (2 * ne)
    on2de = []
    for des in n2de:
        des = des.copy()
        e0 = des.pop(0)
        runs = [[e0]]
        while len(des) > 0:
            current_edge = runs[0][-1]
            next_edge = de2de[current_edge]
            if next_edge == ma.masked:
                next_edge = des.pop(0)
                runs.insert(0, [])
            else:
                run_starts = [run[0] for run in runs]
                if next_edge in run_starts:
                    run_i = run_starts.index(next_edge)
                    runs[0].extend(runs[run_i])
                    runs.pop(run_i)
                    next_edge = des.pop(0)
                    runs.insert(0, [])
                else:
                    des.remove(next_edge)
            runs[0].append(next_edge)
        assert len(runs) <= 2
        if len(runs) == 2:
            edges = runs[0] + runs[1]
        else:
            edges = runs[0]
        on2de.append(edges)
    on2e = [[de % ne for de in run] for run in on2de]
    return [x[::-1] for x in on2e]


def mesh_line_graph(mesh):
    e2n = (
        mesh.edge_node_connectivity.indices_by_location()
        - mesh.edge_node_connectivity.start_index
    )
    ne = e2n.shape[0]
    f2n = (
        mesh.face_node_connectivity.indices_by_location()
        - mesh.face_node_connectivity.start_index
    )
    n2de = node_to_directed_edge(e2n)
    f2de = face_to_deirected_edge(f2n, e2n)
    f2e = face_to_edge(f2de, ne)
    n2e = node_to_ac_edges(n2de, f2de, ne)

    new_node_lon, new_node_lat = mesh.edge_coords
    new_fnc = f2e + n2e
    max_len = max([len(x) for x in new_fnc])
    fnc_array = ma.array(np.zeros([len(new_fnc), max_len], dtype=int), mask=True)
    for i, face in enumerate(new_fnc):
        fnc_array[i, : len(face)] = face
    fnc = Connectivity(fnc_array, cf_role="face_node_connectivity", start_index=0)
    fflon, fflat = mesh.face_coords
    fnlon, fnlat = mesh.node_coords
    flon = np.concatenate([fflon.points, fnlon.points])
    face_lon = AuxCoord(flon, standard_name="longitude")
    flat = np.concatenate([fflat.points, fnlat.points])
    face_lat = AuxCoord(flat, standard_name="latitude")
    new_mesh = Mesh(
        2,
        ((new_node_lon, "x"), (new_node_lat, "y")),
        fnc,
        face_coords_and_axes=((face_lon, "x"), (face_lat, "y")),
    )

    return new_mesh


def convert_edge_cube(cube):
    new_mesh = mesh_line_graph(cube.mesh)
    new_cube = Cube(cube.core_data())
    coords = new_mesh.to_MeshCoords("node")
    for coord in coords:
        new_cube.add_aux_coord(coord, 0)
    return new_cube

def add_edge_centers(mesh):
    lon_coord = mesh.to_MeshCoord("edge", "x")
    lat_coord = mesh.to_MeshCoord("edge", "y")
    lat_mask = (lon_coord.bounds == 90) + (lat_coord.bounds == -90)
    new_lats = np.mean(lat_coord.bounds, axis=-1)

    def lon_mean(lonbds):
        offset = (np.abs(lonbds[:, 0] - lonbds[:, 1]) // 180) * 180
        if ma.is_masked(lonbds):
            offset = offset.filled(0)
        return np.mean(lonbds, axis=-1) + offset
    new_lons = lon_mean(ma.array(lon_coord.bounds, mask=lat_mask))
    mesh.edge_coords.edge_x.points = new_lons
    mesh.edge_coords.edge_y.points = new_lats
    return mesh

