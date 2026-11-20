
Modes of DRP Operation
======================

The SCALES Data Reduction Pipeline (SCALES-DRP) consists of three main
modules: (1) :ref:`Calibration Module-I <calibmodule1>`,
(2) :ref:`Calibration Module-II <calibmodule2>`, and
(3) the :ref:`Science-Grade Module <sci-grade>`.
The :doc:`configuration parameters <Configuration_Parameters>` can be adjusted
to customize the data reduction steps.

An overview of each module and its primary functions is provided below.


.. _calibmodule1:

Calibration Module-I
--------------------

Calibration Module-I automatically processes dark, detector flat,
monochromatic calibration lamp, and bias exposures by applying
:ref:`detector-level corrections <referencepixel>`, and
:ref:`linearity correction <linearity>`, followed by
:ref:`ramp fitting <ramp>` and
:ref:`bad pixel correction <bpmcorrect>`.

The resulting data products for both the IFS and imaging channels
include a bad-pixel-corrected slope image, an associated uncertainty map,
a data quality map, and a chi-square map from the ramp fitting.
These processed exposures are subsequently combined to generate
the corresponding  :ref:`master calibration files <createmaster>` and their associated
uncertainty maps.



.. _calibmodule2:

Calibration Module-II
---------------------

Calibration Module-II generates the calibration reference products
required for :ref:`science-grade data reduction <sci-grade>` and
:ref:`calibration module-I <calibmodule1>`.
These include :ref:`rectification matrices <rmatrix>` and lenslet
flat-field cubes for each IFS observing mode, derived from master
monochromatic calibration exposures; :ref:`read-noise maps <createreadnoise>`
generated from dark exposures; :ref:`linearity coefficients <createlinearity>`,
:ref:`saturation maps <createsat>`, and associated data quality flags
derived from a series of long detector flat exposures; and
:ref:`bad pixel maps <createbpm>` generated from dark and flat-field
slope images.

Unlike routine calibration processing, these reference products
are not necessarily generated on a daily basis. Instead, their
performance and validity are periodically monitored by WMKO staff,
and the products are updated as needed.

.. figure:: /_static/plots/flowchart_calib.png
   :width: 800px
   :align: center

   Flowchart of the SCALES-DRP Calibration Module.


.. _sci-grade:

Science-Grade Module
--------------------

The Science-Grade Module processes science exposures by applying
:ref:`detector-level corrections <referencepixel>`, and
:ref:`linearity correction <linearity>`, followed by
:ref:`ramp fitting <ramp>` and
:ref:`bad pixel correction <bpmcorrect>`.
The resulting slope images then undergo dark subtraction and
detector flat-field correction.

The final data products for both the SCALES-IFS and imaging channels
include a calibrated, bad-pixel-corrected slope image, an associated
uncertainty map, a data quality map, and a chi-square map from the ramp fitting.

For the SCALES-IFS channel, processing continues with
:ref:`spectral extraction <spectralextract>`, producing
three-dimensional IFS data cubes with two spatial dimensions and
one spectral dimension, along with the corresponding uncertainty cubes.



.. figure:: /_static/plots/flowchart_science.png
   :width: 800px
   :align: center

   Flowchart of the SCALES-DRP Science-Grade Module.
