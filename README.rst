SCALES-DRP: The SCALES Data Reduction Pipeline
==============================================
SCALES (Slicer Combined with Array of Lenslets for Exoplanet Spectroscopy)
is a next-generation thermal-infrared coronagraphic imager and integral field
spectrograph (IFS) for the W. M. Keck Observatory. SCALES-DRP is the data
reduction pipeline for SCALES, developed by the SCALES team with support
from the W. M. Keck Observatory.


SCALES-DRP processes raw SCALES exposures to produce calibrated
slope images for both the imager and IFS, and additionally generates
spectrally resolved datacubes for IFS observations.
It is implemented in Python using the Keck DRP framework, supports
parallel processing, and is openly available on GitHub.


Installation
............

The following commands create and activate a ``conda`` environment named
``scalesdrp``:

::

    conda create --name scalesdrp python=3.12
    conda activate scalesdrp

Clone the repository and install the package in editable mode:

::

    git clone https://github.com/scalessim/SCALES-DRP.git
    cd SCALES-DRP
    pip install -e .

Alternatively, install the published package directly from PyPI:

::

    pip install scalesdrp


Quick Start
............

This Quick Start guide explains how to run SCALES-DRP after installation.

The examples assume that the ``scalesdrp`` conda environment has been
created and that ``data_folder`` contains SCALES FITS files, such as
``file_name.fits``. Multiple exposures can also be specified in a text
file (``list.txt``), with one FITS filename per line.


Science-Grade Module
--------------------

The Science-Grade Module processes science exposures using the
configured reduction steps. Processing options can be
customized through the configuration parameters,
defined in ``scalesdrp/configs/scales.cfg``.

Activate the ``scalesdrp`` conda environment and navigate to
the directory containing the SCALES data:

.. code-block:: bash

    conda activate scalesdrp
    cd data_folder

Run the pipeline using one of the following options:

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

The pipeline uses the following output directories and processing table:

* ``redux/``: Stores the reduced data products.

* ``logs/``: Stores log files for the DRP framework
  and SCALES-DRP.

* ``plots/``: Stores diagnostic plots generated during processing.

* ``scales.proc``: Tracks input files and data products.


Calibration Module-I
--------------------

Calibration Module-I processes dark, detector-flat, bias, and monochromatic
calibration exposures to generate master calibration products.
Processing options can be
customized through the configuration parameters,
defined in ``scalesdrp/configs/scales_calib.cfg``.

Activate the ``scalesdrp`` conda environment and navigate to
the directory containing the SCALES data:

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder

Run the pipeline using one of the following options:

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



Output Directories
^^^^^^^^^^^^^^^^^^

The pipeline uses the following output directories and processing table:

* ``redux/``: Stores the reduced data products.

* ``logs/``: Stores log files for the DRP framework
  and SCALES-DRP.

* ``plots/``: Stores diagnostic plots generated during processing.

* ``scales.proc``: Tracks input files and the data products.


Calibration Module-II
----------------------

Calibration Module-II generates supporting detector calibration products
used by Calibration Module-I and the Science-Grade Module.
Processing options can be
customized through the configuration parameters,
defined in ``scalesdrp/configs/scales_calib_sa.cfg``.

Activate the ``scalesdrp`` conda environment and navigate to
the directory containing the SCALES data:

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder

Run the pipeline using one of the following options:

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



Output Directories
^^^^^^^^^^^^^^^^^^

The pipeline uses the following output directories and processing table:

* ``calib_sa/output/``: Stores the reduced data products.

* ``calib_sa/logs/``: Stores log files for the DRP framework
  and SCALES-DRP.

* ``calib_sa/plots/``: Stores diagnostic plots generated during processing.

* ``calib_sa/scales_calib_sa.proc``: Tracks input files and the data products.


Further Documentation
.....................

Detailed documentation covering pipeline architecture, calibration
algorithms, configuration parameters, and advanced usage is available
on the SCALES-DRP Read the Docs website.


Acknowledgements
................

If you use SCALES-DRP in your research, please cite
`Unni et al. (2025) <https://doi.org/10.1117/12.3064542>`_.
