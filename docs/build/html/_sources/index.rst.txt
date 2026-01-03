

Welcome to the SCALES-DRP!
===========================
SCALES (Slicer Combined with Array of Lenslets for Exoplanet Spectroscopy) is a next-generation thermal infrared coronagraphic imager and integral field spectrograph (IFS) for the W. M. Keck Observatory. 
SCALES-DRP is an official data reduction pipeline (DRP) for SCALES. This DRP has been developed by the SCALES team 2024 and supported by the W. M. Keck Observatory.

The SCALES-DRP takes raw SCALES images and turns them into spectrally-dispersed datacubes.

The SCALES-DRP is parallelized, implemented in Python within the Keck-DRP framework, and openly available on `GitHub <https://github.com/scalessim/SCALES-DRP>`_. 

To know more about SCALES
-------------------------
You can find more about SCALES from below publications.

   - `paper1 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12184/121840I/Design-of-SCALES--a-2-5-micron-coronagraphic-integral/10.1117/12.2630577.short>`_ : SCALES overview

   - `paper2 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12184/2630642/SCALES-on-Keck-optical-design/10.1117/12.2630642.short>`_ : SCALES optical design

   
   - `paper3 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12188/121881U/Design-of-an-IR-imaging-channel-for-the-Keck-Observatory/10.1117/12.2630696.short>`_ : Imaging channel 
   
   - `paper4 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11447/1144764/Update-on-the-preliminary-design-of-SCALES--the-Santa/10.1117/12.2562768.short>`_ : Preliminary design of SCALES
   

   - `paper5 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13096/130966W/Performance-analysis-of-SCALES-final-optical-design--end-to/10.1117/12.3019578.short>`_ : Performance analysis of SCALES final optical design: end to end modeling

   - `paper6 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12680/2677831/The-slicer-combined-with-array-of-lenslets-for-exoplanet-spectroscopy/10.1117/12.2677831.short>`_ : Science cases and expected outcome
   
   - `paper7 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11447/114474Z/End-to-end-simulation-of-the-SCALES-integral-field-spectrograph/10.1117/12.2562143.short>`_ : scalessim - A SCALES simulation
   
   - `paper8 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13627/136271L/SCALES-DRP--a-data-reduction-pipeline-for-an-upcoming/10.1117/12.3064542.short>`_ : SCALES-DRP

   - `paper9 <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13627/136271M/Testing-and-characterization-of-monochromator-for-SCALES-calibration-unit/10.1117/12.3065860.short>`_ : SCALES Monochromator calibration unit


Release 1.0
-----------

The latest version of 1.0.0 of the SCALES-DRP includes:

   - Simplified installation via pip and conda environment

   - Vacuum to air and heliocentric or barycentric correction

   - Formal support system via GitHub issues

   - Support for Mac intel M2 and M3 chips in Python 3.12


.. note::

   This project is under active development!



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Quick_Start
   Configuration_Parameters
   Primitives
   Mode_of_Pipelines
   Running_the_SCALES-DRP_Pipeline
   Data_Products
   Support
   Versions
   Updating_Documentation
   FAQ
