import numpy as np

import iris
from iris.cube import Cube
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from esmf_regrid.experimental import util
from esmf_regrid.experimental.unstructured_scheme import (
    regrid_unstructured_to_unstructured,
)


# Load and tweak input meshes

src_fn = "orig_src.nc"
tgt_fn = "orig_tgt.nc"

src_cubes = iris.load(src_fn)
tgt_cubes = iris.load(tgt_fn)

dummy_lon = Cube(np.zeros(10150), standard_name="longitude", var_name="lon2")
dummy_lat = Cube(np.zeros(10150), standard_name="latitude", var_name="lat2")
dummy_face_lon = Cube(np.zeros(5000), standard_name="longitude", var_name="lon3")
dummy_face_lat = Cube(np.zeros(5000), standard_name="latitude", var_name="lat3")
src_cubes.append(dummy_lon)
src_cubes.append(dummy_lat)
src_cubes.append(dummy_face_lon)
src_cubes.append(dummy_face_lat)
src_cubes.extract_cube("latlon").attributes["edge_coordinates"] = "lon2 lat2"
src_cubes.extract_cube("latlon").attributes["face_coordinates"] = "lon3 lat3"
src_cubes.extract_cube("edge_node").attributes["cf_role"] = "edge_node_connectivity"

new_src_fn = "new_src.nc"
iris.save(src_cubes, new_src_fn)

with PARSE_UGRID_ON_LOAD.context():
    new_src_cube = iris.load_cube(new_src_fn)


tgt_cubes = iris.load(tgt_fn)

dummy_lon2 = Cube(np.zeros(3072), standard_name="longitude", var_name="lon2")
dummy_lat2 = Cube(np.zeros(3072), standard_name="latitude", var_name="lat2")
dummy_face_lon2 = Cube(np.zeros(1536), standard_name="longitude", var_name="lon3")
dummy_face_lat2 = Cube(np.zeros(1536), standard_name="latitude", var_name="lat3")
tgt_cubes.append(dummy_lon2)
tgt_cubes.append(dummy_lat2)
tgt_cubes.append(dummy_face_lon2)
tgt_cubes.append(dummy_face_lat2)
main_cube = Cube(
    np.zeros(3072),
    long_name="main",
    attributes={"location": "edge", "mesh": "unit_test"},
)
tgt_cubes.append(main_cube)
tgt_cubes.extract_cube("Topology data of 2D unstructured mesh").attributes[
    "edge_coordinates"
] = "lon2 lat2"
tgt_cubes.extract_cube("Topology data of 2D unstructured mesh").attributes[
    "face_coordinates"
] = "lon3 lat3"

new_tgt_fn = "new_tgt.nc"
iris.save(tgt_cubes, new_tgt_fn)

########################################################################################################################

# Load source meshes and transform into a form we are able to regrid.

with PARSE_UGRID_ON_LOAD.context():
    src_cube = iris.load_cube(new_src_fn)
    tgt_cube = iris.load_cube(new_tgt_fn)

util.add_edge_centers(src_cube.mesh)
util.add_edge_centers(tgt_cube.mesh)

src_dual = util.convert_edge_cube(src_cube)
tgt_dual = util.convert_edge_cube(tgt_cube)

u_src = src_dual.copy()
u_src.data = np.cos(u_src.coord("latitude").points * (np.pi / 180))
v_src = src_dual.copy()
v_src.data = np.sin(u_src.coord("longitude").points * (np.pi / 180))

u_tgt = tgt_dual.copy()
u_tgt.data = np.cos(u_tgt.coord("latitude").points * (np.pi / 180))
v_tgt = tgt_dual.copy()
v_tgt.data = np.sin(u_tgt.coord("longitude").points * (np.pi / 180))

iris.save(u_src, "u_src.nc")
iris.save(v_src, "v_src.nc")
iris.save(u_tgt, "u_tgt.nc")
iris.save(v_tgt, "v_tgt.nc")

# Regrid the data.

u_result = regrid_unstructured_to_unstructured(u_src, tgt_dual, method="bilinear")
v_result = regrid_unstructured_to_unstructured(v_src, tgt_dual, method="bilinear")

u_result_r = regrid_unstructured_to_unstructured(u_tgt, src_dual, method="bilinear")
v_result_r = regrid_unstructured_to_unstructured(v_tgt, src_dual, method="bilinear")

iris.save(u_result, "u_result.nc")
iris.save(v_result, "v_result.nc")
iris.save(u_result_r, "u_result_r.nc")
iris.save(v_result_r, "v_result_r.nc")

print(np.abs(v_result.data - v_tgt.data).max())
print(np.abs(v_result.data - v_tgt.data).mean())
print(np.abs(u_result.data - u_tgt.data).max())
print(np.abs(u_result.data - u_tgt.data).mean())

print(np.abs(v_result_r.data - v_src.data).max())
print(np.abs(v_result_r.data - v_src.data).mean())
print(np.abs(u_result_r.data - u_src.data).max())
print(np.abs(u_result_r.data - u_src.data).mean())
