# iris-esmf-regrid

[![Build Status](https://api.cirrus-ci.com/github/SciTools-incubator/iris-esmf-regrid.svg)](https://cirrus-ci.com/github/SciTools-incubator/iris-esmf-regrid)
[![Documentation Status](https://readthedocs.org/projects/iris-esmf-regrid/badge/?version=latest)](https://iris-esmf-regrid.readthedocs.io/en/latest/?badge=latest)
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

Further documentation can be found [here](https://iris-esmf-regrid.readthedocs.io/en/latest).