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

This project aims to provide a bridge between [Iris](https://github.com/SciTools/iris)
and [ESMF](https://github.com/esmf-org/esmf). This takes the form of regridder classes
which take Iris cubes as their arguments and use ESMF to perform regridding
calculations. These classes are designed to perform well on cubes which have multiple
non-horizontal dimensions and lazy ([Dask](https://github.com/dask/dask)) data.
Both rectilinear and curvilinear grids as well as UGRID meshes have been supported.

## Regridding Example

There are a range of regridder classes (e.g `MeshToGridESMFRegridder` and
`GridToMeshESMFRegridder`). For an example of the regridding process, the
`MeshToGridESMFRegridder` class works as follows:

```python
import iris
from esmf_regrid.experimental.unstructured_scheme import MeshToGridESMFRegridder

source_mesh_cube = iris.load("mesh_cube.nc")
target_grid_cube = iris.load("grid_cube.nc")

# Initialise the regridder with a source mesh and target grid.
regridder = MeshToGridESMFRegridder(source_mesh_cube, target_grid_cube)

# use the initialised regridder to regrid the data from the source cube
# onto a cube with the same grid as `target_grid_cube`.
result = regridder(source_mesh_cube)
```

Note that this pattern allows the reuse of an initialised regridder, saving
significant amounts of time when regridding. To make use of this efficiency across
sessions, we support the saving of certain regridders. We can do this as follows:

```python
from esmf_regrid.experimental.io import load_regridder, save_regridder

# Save the regridder.
save_regridder(regridder, "saved_regridder.nc")

# Load saved regridder.
loaded_regridder = load_regridder("saved_regridder.nc")

# Use loaded regridder.
result = loaded_regridder(source_mesh_cube)
```
