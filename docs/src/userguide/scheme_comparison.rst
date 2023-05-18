Scheme Comparison
=================

There are a number of objects which can be used to regrid Iris_ cubes.
These each have their own quirks which this page aims to describe.

Overview: Schemes
-----------------

The top level objects provided by iris-esmf-regrid as the *schemes*,
these are designed to behave the same as the *schemes* in
:py:mod:`iris.analysis`. The three schemes iris-esmf-regrid provides
are :class:`~esmf_regrid.schemes.ESMFAreaWeighted`,
:class:`~esmf_regrid.schemes.ESMFBilinear` and
:class:`~esmf_regrid.schemes.ESMFNearest`. These wrap the ESMF_
regrid methods :attr:`~esmpy.api.constants.RegridMethod.CONSERVE`,
:attr:`~esmpy.api.constants.RegridMethod.BILINEAR` or
:attr:`~esmpy.api.constants.RegridMethod.NEAREST_STOD` respectively.
The schemes can be by the pattern::

    result_cube = source_cube.regrid(target_cube, ESMFAreaWeighted())

These schemes are flexible and allow the source or target cube to be
defined on an unstructured mesh.

Overview: Regridders
--------------------

The *regridders* are objects one level down from schemes. A regridder
is a class which is designed to handle the regridding of data from
one specific source to one specific target. *Regridders* are useful
because regridding involves a computationally expensive intitialisation
step which can be avoided whenever a *regridder* is reused.
iris-esmf-regrid provides the regridders
:class:`~esmf_regrid.schemes.ESMFAreaWeightedRegridder`,
:class:`~esmf_regrid.schemes.ESMFBilinearRegridder` and
:class:`~esmf_regrid.schemes.ESMFNearestRegridder` which correspond to
:class:`~esmf_regrid.schemes.ESMFAreaWeighted`,
:class:`~esmf_regrid.schemes.ESMFBilinear` and
:class:`~esmf_regrid.schemes.ESMFNearest` respectively.
These can be initialised either by::

    regridder = ESMFAreaWeightedRegridder(source_cube, target_cube)

or equivalently by::

    regridder = ESMFAreaWeighted().regridder(source_cube, target_cube)

This regridder can then be called by::

    result_cube = regridder(source_cube, target_cube)

which can be reused on any cube defined on the same horizontal
coordinates as ``source_cube``.

There are also the experimental regridders
:class:`~esmf_regrid.experimental.unstructured_scheme.MeshToGridESMFRegridder` and
:class:`~esmf_regrid.experimental.unstructured_scheme.GridToMeshESMFRegridder`.
These were formally the only way to do regridding with a source or
target cube defined on an unstructured mesh. These are less flexible and
require that the source/target be defined on a grid/mesh. Unlike the above
regridders whose method is fixed, these regridders take a ``method`` keyword
of ``conservative``, ``bilinear`` or ``nearest``. While most of the
functionality in these regridders have been ported into the above schemes and
regridders, these remain the only regridders capable of being saved and loaded by
:mod:`esmf_regrid.experimental.io`.


Overview: Miscellaneous Functions
---------------------------------

The functions :func:`~esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`,
:func:`~esmf_regrid.experimental.unstructured_scheme.regrid_unstructured_to_rectilinear` and
:func:`~esmf_regrid.experimental.unstructured_scheme.regrid_rectilinear_to_unstructured`
exist as alternative ways to call the same regridding functionality::

    result = regrid_rectilinear_to_rectilinear(source_cube, target_cube)

This function also has a ``method`` keyword which can be ``conservative``, ``bilinear``
or ``nearest``, with ``conservative`` being the default.

Differences Between Methods
---------------------------

This section is under development, for more details see the
:doc:`API documentation<../_api_generated/modules>`.

.. _Iris: https://github.com/SciTools/iris
.. _ESMF: https://github.com/esmf-org/esmf