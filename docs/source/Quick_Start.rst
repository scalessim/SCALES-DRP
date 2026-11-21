
Quick Start
===========

This Quick Start guide provides instructions for running the
SCALES Data Reduction Pipeline (SCALES-DRP) and verifying that
the pipeline has been successfully installed.

The following examples assume that the user has created a
``conda`` environment named ``scalesdrp`` and has a directory
(``data_folder``) containing SCALES FITS files
(e.g., ``file_name.fits``). To process multiple files, a text
file (``list.txt``) containing the names of the FITS files
to be reduced can be provided.


Science-Grade Module
--------------------

The Science-Grade Module processes science exposures using the
configured data reduction steps. The processing options can be
customized through the :doc:`Configuration Parameters <Configuration_Parameters>`,
defined in ``scalesdrp/config/scales.cfg``.

First, activate the ``scalesdrp`` conda environment and navigate
to the directory containing the SCALES data:

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder

The pipeline can be executed using any of the following options.

**1. Process a single FITS file**

Use the ``-f`` option to process an individual FITS file:

.. code-block:: bash

   start_scales_reduce -f file_name.fits

**2. Process multiple FITS files from a list**

Use the ``-l`` option to process multiple FITS files specified
in a text file (``list.txt``), with one filename per line:

.. code-block:: bash

   start_scales_reduce -l list.txt

**3. Process all FITS files in a directory**

Use the ``-d`` option to process FITS files located in a
specified directory:

.. code-block:: bash

   start_scales_reduce -d /full/path/to/data_folder


Output Directories
^^^^^^^^^^^^^^^^^^

When the pipeline is executed, the following output directories
and processing files are created:

* ``redux/``: Contains the reduced data products generated
  by the pipeline.

* ``logs/``: Contains separate log files for the DRP framework
  and the SCALES-DRP pipeline.

* ``plots/``: Contains diagnostic plots generated during
  data reduction.

* ``scales.proc``: A processing table used to track the
  status of input files and their corresponding data products.


Calibration Module-I
--------------------

The Calibration Module-I processes dark, detector flat, bias, and monochromatic
calibration exposures to create the master calibration files. The processing options can be
customized through the :doc:`Configuration Parameters <Configuration_Parameters>`,
defined in ``scalesdrp/config/scales_calib.cfg``.

First, activate the ``scalesdrp`` conda environment and navigate
to the directory containing the SCALES data:

.. code-block:: bash

    conda activate scalesdrp
    cd data_folder

The pipeline can be executed using any of the following options.

**1. Process multiple FITS files from a list**

Use the ``-l`` option to process multiple FITS files specified
in a text file (``list.txt``), with one filename per line:

.. code-block:: bash

    start_scales_calib -l list.txt

**2. Process all FITS files in a directory**

Use the ``-d`` option to process FITS files located in a
specified directory:

.. code-block:: bash

    start_scales_calib -d /full/path/to/data_folder


.. note::

    The ``-f`` option is not recommended for calibration processing,
    as generating a master calibration file generally requires
    combining multiple input FITS files. Instead, use the ``-l``
    or ``-d`` option to process multiple calibration exposures.


Output Directories
^^^^^^^^^^^^^^^^^^

When the pipeline is executed, the following output directories
and processing files are created:

* ``redux/``: Contains the reduced data products generated
  by the pipeline.

* ``logs/``: Contains separate log files for the DRP framework
  and the SCALES-DRP pipeline.

* ``plots/``: Contains diagnostic plots generated during
  data reduction.

* ``scales.proc``: A processing table used to track the
  status of input files and their corresponding data products.


Calibration Module-II
----------------------

The Calibration Module-II produces all the supportive calibration files
to process Calibration Module-I and Science-grade Module. The processing options can be
customized through the :doc:`Configuration Parameters <Configuration_Parameters>`,
defined in ``scalesdrp/config/scales_calib_sa.cfg``.

First, activate the ``scalesdrp`` conda environment and navigate
to the directory containing the SCALES data:

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder

The pipeline can be executed using any of the following options.

**1. Process multiple FITS files from a list**

Use the ``-l`` option to process multiple FITS files specified
in a text file (``list.txt``), with one filename per line:

.. code-block:: bash

   start_scales_calib_sa -l list.txt

**2. Process all FITS files in a directory**

Use the ``-d`` option to process FITS files located in a
specified directory:

.. code-block:: bash

   start_scales_calib_sa -d /full/path/to/data_folder


.. note::

   The ``-f`` option is not recommended for calibration processing,
   as generating all the calibration files generally requires
   combining multiple input FITS files. Instead, use the ``-l``
   or ``-d`` option to process multiple calibration exposures.


Output Directories
^^^^^^^^^^^^^^^^^^

When the pipeline is executed, the following output directories
and processing files are created:

* ``calib_sa/output/``: Contains the reduced data products generated
  by the pipeline.

* ``calib_sa/logs/``: Contains separate log files for the DRP framework
  and the SCALES-DRP pipeline.

* ``calib_sa/plots/``: Contains diagnostic plots generated during
  data reduction.

* ``calib_sa/scales.proc``: A processing table used to track the
  status of input files and their corresponding data products.


.. _othercommand:

Other command line options
--------------------------

.. csv-table:: Usable command tips
   :header: "Symbol", "Meaning", "Note"
   :widths: 15, 30, 30
   :file: tables/table3.csv
