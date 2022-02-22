
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased]

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
