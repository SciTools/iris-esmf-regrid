|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to iris-esmf-regrid for this release
(:doc:`View all changes <index>`.)


.. dropdown:: :opticon:`report` Release Highlights
   :container: + shadow
   :title: text-primary text-center font-weight-bold
   :body: bg-light
   :animate: fade-in
   :open:

   The highlights for this minor release of iris-esmf-regrid include:

   * Support for lazy regridding.

   And finally, get in touch with us on `GitHub`_ if you have any issues or
   feature requests for improving iris-esmf-regrid. Enjoy!


📢 Announcements
================

#. N/A


✨ Features
===========

#. N/A


🐛 Bugs Fixed
=============

#. N/A


💣 Incompatible Changes
=======================

#. N/A


🚀 Performance Enhancements
===========================

#. `@stephenworsley`_ added support for lazy regridding, this showed an
   improvement in the performance of the regridding benchmarks. (:pull:`80`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. N/A


💼 Internal
===========

#. `@trexfeathers`_ added support for ASV benchmarks on the CI. (:pull:`79`)
#. `@lbdreyer`_ fixed an issue with directory naming. (:pull:`92`)
#. `@stephenworsley`_ added benchmarks for regridding with lazy (:pull:`100`)
   and realised data. (:pull:`98`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:

.. _GitHub: https://github.com/SciTools-incubator/iris-esmf-regrid/issues/new/choose
.. _@lbdreyer: https://github.com/lbdreyer
.. _@stephenworsley: https://github.com/stephenworsley
.. _@trexfeathers: https://github.com/trexfeathers