SCALES-DRP: The SCALES Data Reduction Pipeline
==============================================
SCALES (Slicer Combined with Array of Lenslets for Exoplanet Spectroscopy) is a next-generation thermal infrared coronagraphic imager and integral field spectrograph (IFS) for the W. M. Keck Observatory. SCALES-DRP is an official data reduction pipeline (DRP) for SCALES. This DRP has been developed by the SCALES team 2024 and supported by the W. M. Keck Observatory.

The SCALES-DRP takes raw SCALES images and turns them into spectrally-dispersed datacubes.

The SCALES-DRP is parallelized, implemented in Python within the Keck-DRP framework, and openly available on GitHub.

Installation
............

The following steps will create and activate a conda environment named
``scalesdrp``:

::

    conda create --name scalesdrp python=3.12
    conda activate scalesdrp

Clone the repository and install the package in editable mode:

::

    git clone https://github.com/scalessim/SCALES-DRP.git
    cd SCALES-DRP
    pip install -e .

Alternatively, you can install the released version directly from PyPI:

::

    pip install scalesdrp


Running the Pipeline
....................

The SCALES-DRP consists of three main modules:

1. **Calibration module**
2. **Quicklook module**
3. **Science-grade module**


Quicklook Module
................

The quicklook module is designed for rapid, automated execution during
observations. It allows users to interactively visualize individual
exposures and final 3D IFS data cubes using graphical user interfaces
(GUIs) in near real time.

This module continuously searches for newly arrived files in a specified
directory and performs minimal data processing within a fraction of a
second.

- **Imaging mode**: produces a bad-pixel-corrected slope image.
- **IFS mode**: produces both a bad-pixel-corrected slope image and an
  optimally extracted 3D IFS cube for science input data.

To execute the quicklook module:

::

    start_scales_quicklook -d FULL_PATH

Here, ``-d`` specifies the directory containing the input data.

This module creates the following output directories:

- ``redux_ql/`` — quicklook reduction outputs
- ``log/`` — log files containing execution details


Calibration Module
..................

The calibration module is the core of SCALES-DRP. It automatically processes
daily afternoon calibration data required for science-grade data reduction.

This module performs detailed processing for:

- Darks
- Bias reads
- Detector flats
- Lenslet flats
- Monochromatic calibration data

The outputs include:

- Master calibration files for darks, bias, and detector flats (for both
  Imaging and IFS modes)
- Wavelength-binned master files from monochromatic calibrations
- Rectification matrices required by the science-grade pipeline
- Lenslet flat master files and lenslet flat cubes

To execute the calibration module:

::

    start_scales_calib -d FULL_PATH

Here, ``-d`` specifies the directory containing the calibration data.

This module creates the following directories:

- ``redux/`` — calibration reduction outputs
- ``log/`` — detailed log files
- ``plot/`` — diagnostic and quality-control plots


Science-Grade Module
....................

The science-grade module performs the most detailed data reduction on
science observations.

- **Imaging mode**: produces a fully corrected slope image with all
  detector-level corrections applied.
- **IFS mode**: produces a fully reduced 3D IFS cube using one or more
  spectral extraction methods.

For all outputs, associated uncertainty maps and data-quality masks are
also generated.

To run the science-grade module on a single file
(make sure you are in the data directory):

::

    start_scales_reduce -f filename.fits

To process all files in a directory:

::

    start_scales_reduce -d FULL_PATH

To process a specific list of files
(make sure you are in the data directory):

::

    start_scales_reduce -l list.txt

The science-grad module creates the following directories:

- ``redux/`` — calibration reduction outputs
- ``log/`` — detailed log files
- ``plot/`` — diagnostic and quality-control plots

Further Documentation
.....................

More detailed documentation, including pipeline architecture, algorithms,
and advanced usage, can be found on the SCALES Read the Docs page.
