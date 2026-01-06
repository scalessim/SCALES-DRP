Modes of DRP Operation & the Execution
======================================

The DRP consists of three main modules: (1) a calibration module, (2) a quicklook module, and (3) a science grade module. More details and the :doc:`primitives <Primitives>` included in each modules are given below. 

.. _quicklookmodule:

Quicklook Pipeline
------------------

A quicklook module is used for real-time analysis during observations and performs several key steps: :ref:`referencepixel`, :ref:`onebyf` followed by a pixel by pixel ramp fitting using a first-order polynomial.  A rectification based :ref:`bpm` correction is implemented for fast exection. 

For imaging mode, quicklook will automatically display the raw image and the final bad pixel corrected slope image in leass than a second. IFS mode also displays a 3D datacube after a weighted :ref:`optimalextract` using the existing :ref:`rmatrix` created using the :ref:`calibmodule`. 

The quicklook module is designed for automatic rapid execution, which allows users to interactively visualize individual exposures and the final 3D IFS data cubes using GUIs during the observation itself. Usually the execution takes 2 or 3 seconds to produce the datacube using the weighted ``Optimal extraction`` from raw reads.


The quicklook pipeline will search for newly arrived FITS files in the folder and try to exectute based on the FITS header information. Once the analysis is completed, it will wait for the next FITS file to arrive.

A user can access the ``Quicklook Pipeline`` using the following command. 

.. code-block:: shell

   start_scales_quicklook -m -W -d FULL PATH

Here ``-m`` indicate monitoring, ``-W`` indicates wait for ever, and ``-d`` for directory. More details can seen :ref:`here <othercommand>`.

.. _calibmodule:

Calibration Pipeline
--------------------

The calibration module is the core of SCALES-DRP, automatically processing daily afternoon calibration data required for science data processing. Calib module will do a detailed data processing for dark, detector flat, lenslet flat, monochromatic calibration, and bias reads explained in section :doc:`primitives <Primitives>`. 

Master calibration files are created and stored after :ref:`bad pixel correction <bpm>`. For lenslet flat exposures, a additional weighted :ref:`optimal extraction extraction <optimalextract>` is performed to create the lenslet flat cube. This module also produces a bad pixel mask (BPM) using a series of flat and dark exposures, estimates the read-noise map from a set of bias exposures.

The monochromator data will be taken more intermittently as it is expected to remain stable over much longer timescales. The calibration module generates :ref:`rectification matrices <rmatrix>` (RM) for :ref:`spectral extraction <spectralextract>`, and :ref:`wavelength calibration <wave>` using a series of monochromatic calibration lamp exposures. The rectification matrices encode the information of each lenslet PSF (or psflet) across all wavelengths.

Currently, the full calibration routine takes on the order of minutes to execute on a Mac M3 (128GB) laptop. 

.. figure:: /_static/plots/calib_module_flowchart.png
   :width: 800px
   :align: center

   A flowchart of the SCLAES-DRP calibration module.

A user can execute the ``Calibration Pipeline`` using the following command:


.. code-block:: shell

   start_scales_calib -d FULL PATH

where ``-d`` stands for directory follwed by the full path to the directory with calibration files.

.. Note ::
   
   - Calibration module should execute before the science-grad module operation. So that all the required calibration files will be readly avaialble for the execution. 

   - Start calibration module execution **ONLY** after completing the afternoon calibration. 



Science-Grad Pipeline
---------------------
The science module performs detailed detector level corrections explained in section :doc:`primitives <Primitives>`, :ref:`ramp fitting <ramp>`, :ref:`bad pixel correction <bpm>`, and :ref:`spectral extraction <spectralextract>` of the science reads, using outputs from the :ref:`calibration module <calibmodule>`  to generate the wavelength-calibrated 3D IFS datacube. 

.. figure:: /_static/plots/science_module_flowchart.png
   :width: 800px
   :align: center

   A flowchart of the SCLAES-DRP science-grad module.


A user can execute the ``science-grad pipeline`` using the following command:

-  For execting a single FITS file (make sure to execute inside the directory.):

   
   .. code-block:: shell

      start_scales_calib -f filenmae.fits
   
   Here ``-f`` stands for a file.

-  For execting a list of FITS file (make sure to execute inside the directory.):
   
   .. code-block:: shell

      start_scales_calib -l list.txt

   Here ``-l`` stands for a text file with a list of FITS file name to process.

-  For execting for all FITS files in a directory:
   
   .. code-block:: shell

      start_scales_calib -d FULL PATH

   Here ``-d`` stands for a the directory.



