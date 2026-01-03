Primitives
==========

This section gives a brief idea about all the primitives used for **SCALES-DRP** pipeline.
For Imaging mode, the final output is the ramp-fitted data (2048x2048) and the corresponding error image, and different data quality flags used. For IFS mode, the final output is a datacube
with two spatial and one spectral axis. Below we explain the individual steps included in the reduction process in the order of execution.

The SCALES  H2RG detector has four readout channel with both ``slow`` and ``fast`` readout modes.
A odd-even column swapping is required for the ``fast`` mode of observation, which is executed by the detector server as individual read completes. The raw input read to the DRP is odd-even swapped. 

All the primitive files are exist in:

.. code-block:: bash

   /SCALES-DRP/scalesdrp/primitives


.. contents::
   :local:
   :depth: 2

.. _referencepixel:

Reference Pixel Correction
--------------------------
   **Matches all amplifier outputs of the detector to a common level.**

    The primitive subtracts the average of the top and bottom four reference rows
    for each amplifier and individual reads. Calculate the sigma clipped mean value of these reference pixels and subtract it from the each amplifier channel for odd and even column separately. The default average method is ``mean`` but can be changed to ``median`` as well.  There is an option to turn the odd/even step off and replace with a single sigma-clipped mean value for all horizontal reference pixels in each amplifier. And the correction is applied to the (2048x2048) image frame including the top-bottom and left-right reference pixels. 


.. _onebyf:

1/f Noise Correction
--------------------

   **Determines 1/f noise and and subtracts it from the image.**

   The primitive perform a 1/f correction using the left and right four reference pixel columns. Performs an optimal filtering of the vertical reference pixel to reduce 1/f noise (horizontal stripes). A sigma clipped mean value of the reference pixels are estimated estimated across the reads and then global mean value is subtracted from each reference pixels. The residual values are averaged to a single (1,2048) reference pixels and smooths
   using FFT (Fast Fourier Transform) and subtracted from the entire data including the top-bottom and left-right reference pixels. The default average method is ``mean`` but can be changed to ``median`` as well.  FFT method adapted from Kosarev & Pantos algorithm. This assumes that the data to be filtered/smoothed has been sampled evenly. M. Robberto `IDL code  <http://www.stsci.edu/~robberto/Main/Software/IDL4pipeline/>`_. Majority of the python version of the code is adopted from `Jarron Leisenring <https://github.com/JarronL/hxrg_ref_pixels/tree/main>`_.

.. _linearity:

Linearity Correction
--------------------
   **Correct per pixel non-linearity using a pre comupted set of coefficients for individual pixels**

   Linearity correction has two part, creating linearity coefficients for each detector and then applying these coefficients to an input data. A data quality flags (DQ) are created as a by product of linearity correction and will be modified and used in each step of data processing. Let us go through one by one.


Creating linearity coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   **Per-pixel linearity characterization using a detector flat ramp**
      We use a detector flat with enough number of reads covering the linear part, non-linear part, and the saturation of each pixel. 
    

    For each pixel:
      - Detect where the ramp stops increasing (saturation / plateau) using derivatives and a sliding window.
      - Discard saturated reads; keep only pre-saturation data.
      - Fit a provisional linear baseline to a very low-signal core.
      - Use this provisional line to compute fractional deviation of all pre-saturation points, and find the first read where the deviation exceeds `baseline_dev_threshold`.
      - Define the final baseline region as the reads up to `cutoff`, and refit a line there.
      - Use this final baseline line to define a linearized signal L(t).
      - Split (M, L) into two regions using a deviation-based cutoff in L, with a fallback based on  `cutoff_fraction` of the sequence length.
      - Fit two polynomials
            - COEFFS1: low-signal (M -> L), order = low_poly_order (order=1)
            - COEFFS2: high-signal (M -> L), order = high_poly_order (order=3)


    Output FITS HDUs:
      - COEFFS1        (low_poly_order+1, H, W)
      - COEFFS2        (high_poly_order+1, H, W)
      - CUTOFFS        (H, W)  cutoff in linearized DN #switch btw polynomial 1&2
      - SATURATION     (H, W)  estimated saturation level in measured DN
      - SLOPE          (H, W)  baseline linear slope
      - INTERCEPT      (H, W)  baseline linear intercept
      - GOODPIX        (H, W)  1 if coefficients are considered valid

    The linearity coefficient files are not created on a daily basis. The Keck observatory will do a frequent check on the quality of the detector behaviour and update the files as needed. The files are exist in:

    .. code-block:: bash

      /SCALES-DRP/scalesdrp/calib/

Applying linearity coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   **Apply per-pixel two-segment polynomial linearity correction.**

    Use the informations COEFFS1, COEFFS2, CUTOFFS, and GOODPIX for each pixel from the coefficients created for the linearity correction.


    Behaviour
    ---------
    * CUTOFFS is used as a threshold in measured DN to decide between
      low vs high polynomial.

    * Pixels with no usable low segment (COEFFS1 all-NaN, or cutoff NaN,
      or GOODPIX==False) are:

        - flagged with DQ_FLAGS["NO_LIN_CORR"] in pixel_dq
        - forced to use identity y = x for both segments (no correction).

    * Pixels where the high segment looks bad (huge COEFFS2 or COEFFS2
      all-NaN) but the low segment is fine:

        - are NOT flagged NO_LIN_CORR
        - simply never use COEFFS2; they use COEFFS1 at all counts.

    * Saturation cutoff is updated on the corrected ramp.

Data quality flags (DQ flags)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   **Create DQ flags helpful for data processing**

   As part of creating linerity coefficients, we also creating different DQ flags and saved to the output as a FITS extension for each pixel. The DQ flags include

   - **Saturation cutoff:** All pixel above the saturation DN values are flagged as saturated.
   - **Bad pixel flag:** All kind of bad pixels are flagged to 1 refer.
   - **GOODPIX:** All the pixels where the linearity correction is successfully applied.

.. _bpm:

Bad pixel
--------------------
    The bad pixel correction has two part, creating a bad pixel mask using a set of dark and flat exposures and applying this BPM correction to the input data. The BPMs are not created on a daily basis. The Keck observatory will do a frequent check on the quality of the detector behaviour and update the files as needed. The files are exist in:

    .. code-block:: bash

      /SCALES-DRP/scalesdrp/calib/


Creating a BPM
~~~~~~~~~~~~~~
    A bad pixel mask (BPM) is generated by identifying pixels that behave abnormally in either time or space. Unstable pixels are found using a temporal criterion: any pixel whose signal fluctuates by more than five sigma across the stack of exposures is flagged. Static hot or cold pixels are found using a spatial criterion: in each individual frame, a pixel is flagged if it deviates by more than five sigma from the median value of its local neighbors, calculated within a 5×5 kernel. The final BPM combines all pixels flagged by either method, which currently totals 1.58% of the imager detector and 1.58% of the IFS detector.
    

Correcting for bad pixels
~~~~~~~~~~~~~~~~~~~~~~~~~
     A bad pixel correction is applied to individual exposures using two approaches. The first is an iterative median filter of kernel size varying from (3 × 3) to (11 × 11) with a minimum of 0.45% of good neighbors around each bad pixel. The second method is applying an interpolation using a 2D Gaussian kernel with a standard deviation of three sigma. 

Readnoise map
-------------
    Read noise map is created using a set of bad pixel corrected bias exposures averaging after removing the DC offset.

    The readnoise maps are not created on a daily basis. The Keck observatory will do a frequent check on the quality of the detector behaviour and update the files as needed. Readnoise map is important for ramp fitting. The readnoise maps for IFS and imager are exist in:

    .. code-block:: bash

      /SCALES-DRP/scalesdrp/calib/

.. _rmatrix:

Rectification matrix
--------------------------------
    The SCALES calibration unit includes a monochromator capable of producing selectable, narrowband illumination at any central wavelength between 1 − 5μm. Each monochromatic lamp exposure produces a 2D image containing the point-spread functions (PSFs) from all lenslets, referred to as psflets. The rectification matrix (RM) is constructed empirically from these series of monochromatic calibration lamp exposures that densely sample the instrument’s 2 − 5μm wavelength range. This rectification matrices are important for IFS spectral extraction (refer :ref:`spectralextract`) and wavelength calibration. 

    The RM is a linear operator that forms the core of the instrumental model, mapping individual psflets across all wavelengths. This mapping is inherently sparse, since each monochromatic PSF is spatially localized and illuminates approximately a 3×3 box of detector pixels. The RM has dimensions 4194304, Xlenslet ×Ylenslet ×λbin, where 4194304 corresponds to the total number of pixels in the 2048 × 2048 detector, and Xlenslet and Ylenslet represent the number of lenslets in the x and y directions, respectively (approximately 108×108 for low resolution and 17 × 18 for medium resolution). 

    Each column in the rectification matrix represents the weight of each pixel at each wavelength, measured from a calibration unit dataset. As a forward model, the RM transforms a physically meaningful 3D data cube (Acube) into its corresponding 2D raw image (dsim) via the linear operation dsim = R ∗ Acube. Conversely, spectral extraction is formulated as the inverse problem: recovering the optimal three-dimensional data cube (Acube) that, when projected through the RM, best reproduces the observed 2D image (See :ref:`spectralextract` for more details). 

    The Keck observatory will do a frequent check on the quality of the detector behaviour and update the files as needed. A cross-corelation will perform to incoperate the possible shift the psflet location on a daily basis. The rectification matrices for different IFS modes are exist in:

    .. code-block:: bash

      /SCALES-DRP/scalesdrp/calib/

.. _ramp:

Ramp fitting
------------
   **Generate an exposure from a 'N' number of reads or from a group of reads.**

   We adopt the ramp fitting method - ``fitramp`` by  `Brandt et. al. 2024 <https://github.com/t-brandt/fitramp/tree/main>`_ for ramp fitting. This method perform an optimal fit to a pixel’s count rate nondestructively in the presence of both read and photon noise. The method construct a covarience matrix by estimating the difference in the read in a ramp, propagation of the read noise, photon noise and their corelation. And Performs a generalized least squares fit to the differences, using the inverse of the covariance matrix as weights. This gives optimal weight to each difference. The readnoise per pixel is estimated from the ``bias frames``. The jumps are detected iteratively checking the goodness of fit at each possible jump location. More details are presented in `paper1 <https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9/pdf>`_ and 
   `paper2 <https://iopscience.iop.org/article/10.1088/1538-3873/ad38da>`_. 

   We use the saturation mask from the DQ flags to avoid saturated pixels for ramp fitting. 

   The ramp fitting primitive produces images—slope (countrate) and pedestal (reset) using ``fitramp``. Ramp fitting is the final step for imaging mode after bias, dark, and detector flat correction.

   - ``fitramp`` fall back to a robust OLS if ``fitramp`` can’t run for a pixel. 
   - Mask physically impossible read pairs and detect jumps.
    
   Ramp fitting returns the slope image, associated uncertainty, and the reset value estimated and stored as output FITS extension. As part of this primitive BPM correction, bias or dark subtraction, and detector flat fielding is performed depends on the input datatype.
 

Bias and Dark Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~
    Master bias and master dark files are created during the afternoon calibration process. As part of ramp fitting, the primitive will search for these master files and will subtract it from the input data. More details are included in :doc:`modes of DRP execution <Mode_of_Pipelines>`. 

Flat Fielding
~~~~~~~~~~~~~
    Master detector flat and master lenslet flat cube are created during the afternoon calibration process. As part of ramp fitting, the primitive will search for these master files and will divide it from the input data. More details are included in :doc:`modes of DRP execution <Mode_of_Pipelines>`.

.. _spectralextract:

Spectral Extraction
-------------------
    The next general step from ramp image to the final 3D IFS datacube is a spectral extraction after detector flat fielding. The SCALES-DRP implements two main extraction methods, ``optimal extraction`` and ``χ2 extraction``. Both methods produce a 3D IFS datacube along with a corresponding flux error cube, with dimensions defined by the number of lenslets in each spatial direction and the number of wavelength bins along the spectral axis. 

.. _optimalextract:

Optimal Extraction
~~~~~~~~~~~~~~~~~~

   The pipeline implements the standard optimal extraction algorithm by `Horne <https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract>`_. This technique estimates the spectral intensity at each wavelength by weighting the pixels according to the measured line-spread function of the calibration lamp psflet and the associated pixel specific uncertainties. We implemented optimal extraction using a linear algebraic rectification matrix generated using the :doc:`calibration module <Mode_of_Pipelines>`. Each column in the rectification matrix represents the weight of each pixel at each wavelength, measured from a monochromatic calibration lamp dataset. A total variance map is computed for each pixel by combining contributions from photon noise and read noise, ensuring proper weighting of the data during the optimal extraction process. The output errorcube is the error propagation followed in each step of the optimial extraction process.
   

.. _leastextract:

Least square Extraction
~~~~~~~~~~~~~~~~~~~~~~~

   The χ2 based spectral extraction differs from optimal extraction by fitting the entire two-dimensional data at predefined psflet positions using a rectification matrix of shape (4194304, Xlenslet ×Ylenslet ×λbin ). Each column in the rectification matrix represents the weight of each pixel at each wavelength, measured from a calibration unit dataset. More details about rectification matrix is presented in the :doc:`calibration module <Mode_of_Pipelines>`.  The one-dimensional optimal flux from each psflet is extracted using a forward-modeling technique that formulates the extraction as a linear inverse problem, ``R × A = d``, where R is the rectification matrix, A is the best-fit flux for each psflet as a function of wavelength, and d is the observed 2D detector image. The optimal solution for A is found by minimizing the chi-squared statistic, which represents the variance-weighted sum of squared residuals between the forward-model and the input data. The weights are derived from both detector read noise and signal-dependent photon noise. To ensure a physically plausible result, the optimization is subject to two constraints: a non-negativity constraint on the flux (A ≥ 0), and a dynamic upper bound derived from optimal extraction. his bounded, weighted, non-negative least-squares (BWNNLS) system is solved efficiently using a trust-region reflective algorithm, as implemented in ``scipy.optimize.lsq-linear``.

.. _wave:

Wavelevngth Calibration
------------------------
    The spectral resolution of each  mode of observation is depends on the number of monochromatic calibration exposures are used to create the rectification matrices (See :ref:`rmatrix` for more details).  Currently no interpolation is added to the final wavelength solution.

.. _wcs:

World Coordinate System (WCS)
-----------------------------
    This function constructs a three-dimensional World Coordinate System (WCS) for a generic integral field unit (IFU) data cube. It is designed to be flexible and robust to variations in FITS header conventions by automatically parsing sky coordinates provided in different formats, handling optional or missing keywords, and applying sensible defaults where needed. The resulting WCS maps the spatial dimensions of the cube to right ascension and declination using a tangent-plane projection, and the spectral dimension to wavelength, incorporating the correct pixel scale and detector position angle.

    The function takes as input the shape of the data cube, the FITS header containing observational metadata, a dictionary that maps filter names to their corresponding reference pixel positions on the detector, and a default reference pixel to use when no filter-specific entry is available. Using this information, it determines the reference sky coordinates, spatial and spectral scaling, detector orientation, and reference pixel locations, and returns a fully constructed ``astropy.wcs.WCS`` object suitable for accurate spatial and spectral coordinate transformations.


















