# iris-esmf-regrid

[![Build Status](https://api.cirrus-ci.com/github/SciTools-incubator/iris-esmf-regrid.svg)](https://cirrus-ci.com/github/SciTools-incubator/iris-esmf-regrid)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SciTools-incubator/iris-esmf-regrid/main.svg)](https://results.pre-commit.ci/latest/github/SciTools-incubator/iris-esmf-regrid/master)
[![codecov](https://codecov.io/gh/SciTools-incubator/iris-esmf-regrid/branch/main/graph/badge.svg?token=PKBXEHOZFT)](https://codecov.io/gh/SciTools-incubator/iris-esmf-regrid)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/SciTools-incubator/iris-esmf-regrid)](https://github.com/SciTools-incubator/iris-esmf-regrid/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/SciTools-incubator/iris-esmf-regrid)](https://github.com/SciTools-incubator/iris-esmf-regrid/graphs/contributors)
![Mark stale issues and pull requests](https://github.com/SciTools-incubator/iris-esmf-regrid/workflows/Mark%20stale%20issues%20and%20pull%20requests/badge.svg)

---

## Overview

This project aims to provide a bridge between [iris](https://github.com/SciTools/iris)
and [ESMF](https://github.com/esmf-org/esmf). This takes the form of regridder classes
which take iris cubes as their arguments and use ESMF to perform regridding
calculations. These classes are designed to perform well on cubes which have multiple
non-horizontal dimensions and lazy ([dask](https://github.com/dask/dask)) data.
Both rectilinear and curvilinear grids as well ass UGRID meshes have been supported
(though not all combinations of these are supported yet).

## Example

There are currently three regridder classes: `ESMFAreaWeightedRegridder`,
`MeshToGridESMFRegridder` and `GridToMeshESMFRegridder`. The first of these works
on cubes based on logically rectangular grids, the others are designed for the case
where either the source or target cube is based on a UGRID type mesh. Except for
the types of cubes that these take as arguments, all three of these work in essentially
the same way. We can illustrate how regridding works generally by focusing on the
`MeshToGridESMFRegridder` class:

```python
import iris
from esmf_regrid.experimental.unstructured_scheme import MeshToGridESMFRegridder

source_mesh_cube = iris.load("mesh_cube.nc")
target_grid_cube = iris.load("grid_cube.nc")

regridder = MeshToGridESMFRegridder(source_mesh_cube, target_grid_cube)

result = regridder(source_mesh_cube)
```
