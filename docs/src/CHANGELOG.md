
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
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

