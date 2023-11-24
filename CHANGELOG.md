
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.9] - 2023-11-03

### Added

- [PR#178](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/178)
  Added support for coordinate systems with non-degree type units.
  [@stephenworsley](https://github.com/stephenworsley)

- [PR#311](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/311)
  Added support for Mesh to Mesh regridding.
  [@HGWright](https://github.com/HGWright)

### Fixed

- [PR#301](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/301)
  Fixed a bug which caused errors when regridding with the node locations
  of a mesh whose face_node_connectivity had non-zero start_index.
  [@stephenworsley](https://github.com/stephenworsley)

## [0.8] - 2023-08-22

### Added

- [PR#289](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/289)
  Added the ability to regrid onto a Mesh as a target instead of a Cube.
  [@stephenworsley](https://github.com/stephenworsley)

## [0.7] - 2023-05-23

### Added

- [PR#198](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/198)
  As a result of refactoring regridders to all derive from the same class,
  `_ESMFRegridder`, functionality has been added to the `ESMFAreaWeighted`
  scheme and a new scheme, `ESMFBilinear`, has been added.
  These schemes are now able to handle both grids and meshes.
  Additionally, they are also able to specify the resolution of cells in
  these grids with the `src_resolution` and `tgt_resolution` keywords.
  [@stephenworsley](https://github.com/stephenworsley) with extensive review
  work from [@trexfeathers](https://github.com/trexfeathers)
- [PR#266](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/266)
  Added Nearest neighbour regridding.
  [@stephenworsley](https://github.com/stephenworsley)
  [@HGWright](https://github.com/HGWright)
- [PR#272](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/272)
  Add support for Python 3.11.
  [@stephenworsley](https://github.com/stephenworsley)

### Changed

- [PR#198](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/198)
  Refactor regridders to all derive from the same class `_ESMFRegridder`.
  For the sake of consistency, the resolution keyword in
  `GridToMeshESMFRegridder` and `MeshToGridESMFRegridder` have been
  replaced by `src_resolution` and `tgt_resolution` respectively.
  [@stephenworsley](https://github.com/stephenworsley) with extensive review
  work from [@trexfeathers](https://github.com/trexfeathers)

### Removed

- [PR#272](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/272)
  Remove support for Python 3.8.
  [@stephenworsley](https://github.com/stephenworsley)

### Fixed

- [PR#258](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/258)
  Allow the documentation to build properly.
  [@zklaus](https://github.com/zklaus)

## [0.6] - 2023-03-31

### Added

- [PR#217](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/217)
  Changed the behaviour of coordinate fetching to allow Cubes with both
  1D DimCoords and 2D AuxCoords. In this case the DimCoords are prioritised.
  [@stephenworsley](https://github.com/stephenworsley)
- [PR#220](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/220)
  Matured the benchmarking architecture in line with the latest setup in
  SciTools/iris.
  [@trexfeathers](https://github.com/trexfeathers)
- [PR#241](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/241)
  Fixed compatibility with esmpy 8.4.
  [@stephenworsley](https://github.com/stephenworsley) with help from 
  [@bjlittle](https://github.com/bjlittle) and
  [@valeriupredoi](https://github.com/valeriupredoi)
- [PR#219](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/219)
  Added support for 2D AuxCoords with discontiguities under masked values
  with the use_src_mask and use_tgt_mask keywords.
  [@stephenworsley](https://github.com/stephenworsley)with extensive review
  work from [@trexfeathers](https://github.com/trexfeathers)

### Fixed
- [PR#242](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/242)
  Fixed an issue which caused regridding to curvilinear grids with lazy
  data to fail.
  [@stephenworsley](https://github.com/stephenworsley)

## [0.5] - 2022-10-14

This release improves the support for features such as Bilinear regridding,
curvilinear grids and low resolution grids.

### Added

- [PR#148](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/148)
  Added support for Bilinear regridding for unstructured regridding.
- [PR#165](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/165)
  Added RefinedGridInfo and resolution keyword for unstructured regridders.
- [PR#166](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/166)
  Made weights array handling more robust for different formats of
  pre-computed weights matrices.
- [PR#175](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/175)
  Add curvilinear support for unstructured regridders.
- [PR#208](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/208)
  Unpin Python.
 
## [0.4] - 2022-02-24

This release added the ability to regrid data stored on a UGRID mesh.

### Added

- [PR#31](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/31)
  [PR#32](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/32)
  [PR#36](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/36)
  [PR#39](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/39)
  [PR#46](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/46)
  [PR#55](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/55)
  [PR#96](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/96)
  Added the unstructured regridders `GridToMeshESMFRegridder` and
  `MeshToGridESMFRegridder`.
  [@stephenworsley](https://github.com/stephenworsley) with extensive review
  work from [@abooton](https://github.com/abooton) and
  [@jamesp](https://github.com/jamesp) with benchmarking help from
  [@trexfeathers](https://github.com/trexfeathers)
- [PR#130](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/130)
  [PR#137](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/137)
  Added functions for saving of the unstructured regridders.
  [@stephenworsley](https://github.com/stephenworsley)
- [PR#155](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/155)
  Enabled Sphinx and RTD for automatically rendering the API.
  [@trexfeathers](https://github.com/trexfeathers)

## [0.3] - 2021-12-21

The major change with this version was the addition of the ability to
regrid curvilinear grids (i.e. grids with 2D arrays of coordinates).

### Added
- [PR#125](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/125)
  Added support for curvilinear grids, i.e. cubes with 2D lat/lon coords.
  [@stephenworsley](https://github.com/stephenworsley)
- [PR#124](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/124)
  Improved generation of benchmark data to allow data to be generated from
  a common version/environment.
  [@trexfeathers](https://github.com/trexfeathers)

## [0.2] - 2021-08-25

The major change in this version is the addition of lazy regridding.
This defers the calculation of regridding to the realisation of the data
when the data is a dask array. Calculations may be parallelised via dask.
 
### Added
- [PR#80](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/80)
  Added support for lazy regridding, this showed an improvement in the
  performance of the regridding benchmarks.
  [@stephenworsley](https://github.com/stephenworsley)
- [PR#79](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/79)
  Added support for benchmarks on the CI.
  [@trexfeathers](https://github.com/trexfeathers)
- [PR#98](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/98)
  Added benchmarks for regridding with realised data.
  [@stephenworsley](https://github.com/stephenworsley)
- [PR#100](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/100)
  Added benchmarks for regridding with lazy data.
  [@stephenworsley](https://github.com/stephenworsley)
 
### Fixed
- [PR#92](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/92)
  Fixed an issue with directory naming. [@lbdreyer](https://github.com/lbdreyer)
- [PR#83](https://github.com/SciTools-incubator/iris-esmf-regrid/pull/83)
  Added missing docstrings. [@stephenworsley](https://github.com/stephenworsley)
