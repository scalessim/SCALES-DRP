.. _configparams:

Configuration Parameters
========================

**Controls pipeline execution, detector processing, calibration generation, and spectral extraction.**

The SCALES-DRP provides configurable parameters for enabling or disabling processing steps and selecting calibration products. Three configuration files are located in ``SCALES-DRP/scalesdrp/config/``:

* ``scales.cfg`` -- science-grade reduction of observations and
  generation of calibrated science products.
* ``scales_calib.cfg`` -- calibration module I, which produces master
  darks, biases, detector flats, and processed monochromator exposures.
* ``scales_calib_sa.cfg`` -- calibration module II, which produces linearity coefficients,
  saturation maps, read-noise maps, bad-pixel maps, and rectification matrices.


.. note::

   If the SCALES-DRP was installed using ``pip``, the configuration
   files are stored within the installed Python package directory
   and may not be easily accessible for modification.

   We recommend creating a local copy of the configuration file
   rather than modifying the installed package directly.

   A copy of the configuration file can be generated using the
   ``--write_config`` option:

   .. code-block:: bash

      start_scales_reduce --write_config

      start_scales_calib --write_config

      start_scales_calib_sa --write_config

   The generated configuration file can then be edited to
   customize the pipeline settings.


The tables below follow the supplied configuration reference. **Bold** values
indicate the selections shown in that reference, not necessarily the defaults in every
installed version. File names and calibration availability
should be checked against the local installation.


Common Configuration Parameters
-------------------------------


Amplifier reference-row correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``nchans``
     - **4**
     - Number of detector amplifier readout channels.
   * - ``altcol``
     - **True**, False
     - Enables separate even/odd column reference offsets for each amplifier.
   * - ``channelwise``
     - **True**, False
     - When ``altcol=False``, selects amplifier-wide (True) or column-wise (False) correction.
   * - ``amp_mean_func``
     - **robust.mean**, np.median, np.mean
     - Statistic used to estimate offsets from top and bottom reference rows.

Additional ACN correction
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``do_acn``
     - **True**, False
     - Enables row-dependent even/odd column correction using side reference pixels.
   * - ``acn_avg_type``
     - **pix**, frame, int
     - Reference normalization per pixel, per frame, or over the integration.
   * - ``acn_mean_func``
     - **np.median**, np.mean
     - Statistic used for ACN reference-level estimation.
   * - ``acn_smooth``
     - **True**, False
     - Enables smoothing of row-dependent ACN profiles.
   * - ``acn_savgol``
     - True, **False**
     - Selects Savitzky-Golay (True) or FFT-based (False) smoothing.
   * - ``acn_winsize``
     - **31**
     - Window length for Savitzky-Golay filtering.
   * - ``acn_order``
     - **3**
     - Polynomial order for Savitzky-Golay filtering.
   * - ``resid_colsub``
     - True, **False**
     - Enables additional residual column-wise bias subtraction.

1/f noise correction
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``fixcol``
     - **True**, False
     - Enables row-correlated 1/f noise correction using side reference columns.
   * - ``ref_avg_type``
     - **row_wise**, pix, frame, int
     - Selects the reference-level estimation strategy.
   * - ``ref_mean_func``
     - **np.median**, np.mean
     - Statistic used for reference-level estimation; row-wise mode uses np.nanmedian.
   * - ``ref_smooth``
     - **True**, False
     - Enables smoothing of the 1/f reference profile.
   * - ``ref_savgol``
     - True, **False**
     - Selects Savitzky-Golay (True) or FFT-based (False) filtering.
   * - ``ref_winsize``
     - **31**
     - Window length for Savitzky-Golay filtering.
   * - ``ref_order``
     - **3**
     - Polynomial order for Savitzky-Golay filtering.

Pickup noise correction
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``pickup``
     - True, **False**
     - Enables correction of evolving row-correlated pickup noise.
   * - ``sigma_thresh``
     - **4.0**
     - Sigma threshold for identifying anomalous pixels.
   * - ``dilate_iter``
     - **2**
     - Number of binary dilation iterations used to expand the anomalous-pixel mask.
   * - ``highpass_size``
     - **101**
     - Filter window size used to isolate residual pickup structure.
   * - ``per_amp``
     - True, **False**
     - Selects amplifier-dependent or detector-wide pickup correction.

Linearity correction
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``do_linearity``
     - False, **True**
     - Apply linearity correction to the input ramp

Apply saturation correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``apply_sat_mask``
     - False, **True**
     - Apply saturation map to the linearity corrected input ramp

bad pixel correction
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``apply_bpm``
     - False, **True**
     - Apply bad pixel map to the ramp fitted slope image

Odd-even column swapping
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``do_swap``
     - True, **False**
     - Swap odd and even column of IFS detector during fast readout mode.
       Normally does not require modification.

Rewrite the output
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``clobber``
     - True, **False**
     - Redo the analysis and rewrite the slope image and IFS data cube even though already exist.


Name of the calibration files used for IFS detector. Files are stored in ``SCALES-DRP/scalesdrp/calib/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``bpm_ifs``
     - bpm_ifs.fits
     - Name of the bad pixel map for IFS.
   * - ``bpmat_ifs``
     - bpmat_ifs.npz
     - Name of the bad pixel map for IFS in ``npz`` format.
   * - ``sig_map_ifs_fast0p6``
     - readnoise_ifs_fast0.6.fits
     - Name of the readnoise map for IFS fast0.6 mode
   * - ``flat_ifs_fast0p6``
     - ifs_fast0.6_mflatlamp.fits
     - Name of the default detector flat for IFS fast0.6 mode.
   * - ``lin_coeff_ifs_fast0p6``
     - linearity_ifs_fast0.6.fits
     - Name of the linearity coefficient file for IFS fast0.6 mode.
   * - ``sat_map_ifs_fast0p6``
     - ifs_fast0.6_sat_map.fits
     - Name of the saturation map for IFS fast0.6
   * - ``gain_ifs_fast0p6``
     - 1.0
     - Gain value for IFS fast0.6 mode
   * - ``sig_map_ifs_fast1``
     - readnoise_ifs_fast1.0.fits
     - Name of the readnoise map for IFS fast1.0 mode
   * - ``flat_ifs_fast1``
     - ifs_fast1.0_mflatlamp.fits
     - Name of the default detector flat for IFS fast1.0 mode.
   * - ``lin_coeff_ifs_fast1``
     - linearity_ifs_fast1.0.fits
     - Name of the linearity coefficient file for IFS fast1.0 mode.
   * - ``sat_map_ifs_fast1``
     - ifs_fast1.0_sat_map.fits
     - Name of the saturation map for IFS fast1.0
   * - ``gain_ifs_fast1``
     - 1.0
     - Gain value for IFS fast1.0 mode
   * - ``sig_map_ifs_slow``
     - readnoise_ifs_slow.fits
     - Name of the readnoise map for IFS slow5.2 mode
   * - ``flat_ifs_slow``
     - ifs_slow_mflatlamp.fits
     - Name of the default detector flat for IFS slow5.2 mode.
   * - ``lin_coeff_ifs_slow``
     - linearity_ifs_slow.fits
     - Name of the linearity coefficient file for IFS slow5.2 mode.
   * - ``sat_map_ifs_slow``
     - ifs_slow_sat_map.fits
     - Name of the saturation map for IFS slow5.2
   * - ``gain_ifs_slow``
     - 1.0
     - Gain value for IFS slow5.2 mode

Name of the calibration files used for Imager detector. Files are stored in ``SCALES-DRP/scalesdrp/calib/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``bpm_img``
     - bpm_img.fits
     - Name of the bad pixel map for Imager.
   * - ``bpmat_img``
     - bpmat_img.npz
     - Name of the bad pixel map for Imager in ``npz`` format.
   * - ``sig_map_img_fast0p6``
     - readnoise_img_fast0.6.fits
     - Name of the readnoise map for IMG fast0.6 mode
   * - ``flat_img_fast0p6``
     - img_fast0.6_mflatlamp.fits
     - Name of the default detector flat for IMG fast0.6 mode.
   * - ``lin_coeff_img_fast0p6``
     - linearity_img_fast0.6.fits
     - Name of the linearity coefficient file for IMG fast0.6 mode.
   * - ``sat_map_img_fast0p6``
     - img_fast0.6_sat_map.fits
     - Name of the saturation map for IMG fast0.6
   * - ``gain_img_fast0p6``
     - 1.0
     - Gain value for IMG fast0.6 mode
   * - ``sig_map_img_fast1``
     - readnoise_img_fast1.0.fits
     - Name of the readnoise map for IMG fast1.0 mode
   * - ``flat_img_fast1``
     - img_fast1.0_mflatlamp.fits
     - Name of the default detector flat for IMG fast1.0 mode.
   * - ``lin_coeff_img_fast1``
     - linearity_img_fast1.0.fits
     - Name of the linearity coefficient file for IMG fast1.0 mode.
   * - ``sat_map_img_fast1``
     - img_fast1.0_sat_map.fits
     - Name of the saturation map for IMG fast1.0
   * - ``gain_img_fast1``
     - 1.0
     - Gain value for IMG fast1.0 mode
   * - ``sig_map_img_slow``
     - readnoise_img_slow.fits
     - Name of the readnoise map for IMG slow5.2 mode
   * - ``flat_img_slow``
     - img_slow_mflatlamp.fits
     - Name of the default detector flat for IMG slow5.2 mode.
   * - ``lin_coeff_img_slow``
     - linearity_img_slow.fits
     - Name of the linearity coefficient file for IMG slow5.2 mode.
   * - ``sat_map_img_slow``
     - img_slow_sat_map.fits
     - Name of the saturation map for IMG slow5.2
   * - ``gain_img_slow``
     - 1.0
     - Gain value for IMG slow5.2 mode

Science-Grade Reduction (``scales.cfg``)
-----------------------------------------


Apply various calibration to science image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``apply_dark``
     - **True**, False
     - Enables subtract a master dark slope image from the science slope image matches the exposure time
   * - ``apply_det_flat``
     - **True**, False
     - Enables perform a flat fielding?
   * - ``apply_bias``
     - **True**, False
     - Do you like subtract master bias from an read? (This applied in the case of single read exposures).
   * - ``apply_lens_flat``
     - **True**, False
     - Enables perform a lenslet flat fielding?

Additional Correction
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``subtract_row_median``
     - **False**,True
     - Additional row-wise correction on the bad pixel corrected slope image
   * - ``subtract_col_median``
     - **False**,True
     - Additional column-wise correction on the bad pixel corrected slope image
   * - ``subtract_img_readout_channels``
     - **False**,True
     - Additional channel-wise correction on the bad pixel corrected slope image

IFS spectral Extraction
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``do_chi2_full``
     - **True**, False
     - Enables perform chi-squared-based IFS spectral extraction in addition to the optimal extraction?

Shape of the final IFS datacube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``lowres_lenslet_grid``
     - (112, 112)
     - Number of lenslet in x-direction and number of lenslet in y-direction.
       Change only when using a rectification matrix with different spatial dimensions.
   * - ``medres_lenslet_grid``
     - (17, 18)
     - Number of lenslet in x-direction and number of lenslet in y-direction.
       Change only when using a rectification matrix with different spatial dimensions.

Name of the calibration files used for spectral Extraction. Files are stored in ``SCALES-DRP/scalesdrp/calib/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``OPT_rmat_LowRes_K``
     - LowRes-K_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-K mode.
   * - ``C2_rmat_LowRes_K``
     - LowRes-K_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-K mode.
   * - ``OPT_rmat_LowRes_L``
     - LowRes-L_OPT_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-L mode.
   * - ``C2_rmat_LowRes_L``
     - LowRes-L_C2_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-L mode.
   * - ``OPT_rmat_LowRes_M``
     - LowRes-M_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-M mode.
   * - ``C2_rmat_LowRes_M``
     - LowRes-M_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-M mode.
   * - ``OPT_rmat_LowRes_KLM``
     - LowRes-KLM_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-KLM mode.
   * - ``C2_rmat_LowRes_KLM``
     - LowRes-KLM_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-KLM mode.
   * - ``OPT_rmat_LowRes_KL``
     - LowRes-KL_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-KL mode.
   * - ``C2_rmat_LowRes_KL``
     - LowRes-KL_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-KL mode.
   * - ``OPT_rmat_LowRes_Ls``
     - LowRes-Ls_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in LowRes-Ls mode.
   * - ``C2_rmat_LowRes_Ls``
     - LowRes-Ls_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in LowRes-Ls mode.
   * - ``OPT_rmat_MedRes_K``
     - MedRes-K_OPT_intp_rectmat_260604.npz
     - Name of the rectification matrix for optimal extraction in MedRes-K mode.
   * - ``C2_rmat_MedRes_K``
     - MedRes-K_C2_intp_rectmat_260604.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in MedRes-K mode.
   * - ``OPT_rmat_MedRes_L``
     - MedRes-L_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in MedRes-L mode.
   * - ``C2_rmat_MedRes_L``
     - MedRes-L_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in MedRes-L mode.
   * - ``OPT_rmat_MedRes_M``
     - MedRes-M_OPT_intp_rectmat.npz
     - Name of the rectification matrix for optimal extraction in MedRes-M mode.
   * - ``C2_rmat_MedRes_M``
     - MedRes-M_C2_intp_rectmat.npz
     - Name of the rectification matrix for $chi-squared$-based extraction in MedRes-M mode.

Name of the directories
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``plots``
     - plots
     - The directory in which the diagnostic plots are saved. Normally does not require modification.
   * - ``redux``
     - redux
     - The directory in which the data outputs are saved. Normally does not require modification.
   * - ``logs``
     - logs
     - The directory in which the supportive log files are saved. Normally does not require modification.


Calibration Module I (``scales_calib.cfg``)
--------------------------------------------


Method use to make master calibration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``master_comb_met``
     - **median**, ivw, mean
     - Method use to combine files

Name of the directories
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``plots``
     - plots
     - The directory in which the diagnostic plots are saved. Normally does not require modification.
   * - ``redux``
     - redux
     - The directory in which the data outputs are saved. Normally does not require modification.
   * - ``logs``
     - logs
     - The directory in which the supportive log files are saved. Normally does not require modification.



Calibration Module II (``scales_calib_sa.cfg``)
------------------------------------------------


Create Linearity Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``create_linearity_coeff``
     - False, **True**
     - Enables generation of new linearity coefficients.
   * - ``min_lin_files``
     - 30
     - Minimum number of ramps required to create linearity coefficients
   * - ``min_lin_reads``
     - 50
     - Minimum number of reads required in each ramp

Create Readnoise Map
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``create_readnoise``
     - False, **True**
     - Enables create a new readnoise map?
   * - ``min_rn_reads``
     - 2
     - Minimum reads per ramp to create a readnoise map
   * - ``minimum_rn_ramps``
     - 30
     - Minimum of ramp to create a readnoise map
   * - ``rn_sub_global``
     - False, **True**
     - Enables remove additional global offset from the dark frames during read frames?
   * - ``rn_sub_amp``
     - True, **False**
     - Apply additional column-wise subtraction to each CDS image before estimating noise.
   * - ``rn_sub_row``
     - True, **False**
     - Apply additional row-wise subtraction to each CDS image before estimating noise.

Create Saturation Map
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``create_sat_map``
     - False, **True**
     - Enables create a new saturation map?
   * - ``minimum_detected_ramps``
     - 35
     - Minimum number of ramps need to create saturation map
   * - ``min_sat_reads``
     - 30
     - Minimum number of reads per ramp need to create saturation map
   * - ``drop_fraction``
     - 0.85
     - Fractional difference threshold falls below early reference value
   * - ``minimum_saturation_dn``
     - 1200.0
     - Minimum saturation value threshold in DN.

Create bad pixel Map
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``create_bpm``
     - False, **True**
     - Enables create a new bad pixel map?
   * - ``bpm_min_darks``
     - 3
     - Minimum number of dark slope images available for creating a new bad pixel map.
   * - ``bpm_min_flats``
     - 3
     - Minimum number of detector flat slope images available for creating a new bad pixel map.
   * - ``im_dark_spatial_sigma_thresh``
     - 8
     - Threshold, in units of local robust scatter, for identifying pixels whose signals deviate significantly from their surrounding pixels.
   * - ``im_dark_spatial_kernel_size``
     - 7
     - Size of the square median-filter kernel used to estimate the local background and spatial scatter.
   * - ``im_dark_min_frames_bad_spatial``
     - 4
     - Minimum number of input frames in which a pixel must be identified as a spatial outlier before it is classified as defective.
   * - ``im_dark_row_correction``
     - False, **True**
     - Enables row-wise median subtraction during dark-frame bad-pixel detection to reduce false detections caused by coherent row-dependent signal variations.
   * - ``im_dark_row_suppression_fraction``
     - 0.2
     - Minimum fraction of pixels initially flagged within a row required to suppress candidates that are no longer identified as defective after row correction.
   * - ``im_dark_hot_sigma``
     - 2.5
     - Number of robust standard deviations above the median dark signal used to identify hot pixels.
   * - ``im_dark_cold_sigma``
     - 3.0
     - Number of robust standard deviations below the median dark signal used to identify cold pixels.
   * - ``im_dark_temporal_sigma_thresh``
     - 15.0
     - Threshold, in robust standard deviations above the population median temporal scatter, for identifying unstable pixels. Used when no absolute cutoff is supplied.
   * - ``im_flat_temporal_sigma_thresh``
     - 12.0
     - Threshold, in robust standard deviations above the population median temporal scatter, for identifying unstable pixels. Used when no absolute cutoff is supplied.
   * - ``im_flat_spatial_sigma_thresh``
     - 8.0
     - Threshold, in units of local robust scatter, for identifying pixels whose signals deviate significantly from their surrounding pixels.
   * - ``im_flat_spatial_kernel_size``
     - 7.0
     - Size of the square median-filter kernel used to estimate the local background and spatial scatter.
   * - ``im_flat_min_frames_bad_spatial``
     - 3.0
     - Minimum number of input frames in which a pixel must be identified as a spatial outlier before it is classified as defective.
   * - ``im_normalize_flats``
     - False, **True**
     - Normalizes flat-field images by a locally estimated illumination pattern before identifying pixels with abnormal responses.
   * - ``im_flat_norm_kernel_size``
     - 31
     - Size of the median-filter kernel used to estimate the spatial illumination pattern for flat-field normalization.
   * - ``im_flat_row_correction``
     - False, **True**
     - Removes the median signal of each detector row before calculating temporal scatter, reducing row-correlated variations.
   * - ``im_flat_row_suppression_fraction``
     - 0.20
     - Minimum fraction of pixels initially flagged within a row required to suppress candidates that are no longer identified as defective after row correction.
   * - ``im_flat_low_response_thresh``
     - 0.30
     - Lower threshold for normalized flat-field response. Pixels below this value are classified as low-response (LOW_QE) pixels.
   * - ``im_flat_high_response_thresh``
     - 1.4
     - Upper threshold for normalized flat-field response. Pixels above this value are classified as high-response pixels.
   * - ``ifs_dark_spatial_sigma_thresh``
     - 8.0
     - Threshold, in units of local robust scatter, for identifying pixels whose signals deviate significantly from their surrounding pixels.
   * - ``ifs_dark_spatial_kernel_size``
     - 7.0
     - Size of the square median-filter kernel used to estimate the local background and spatial scatter.
   * - ``ifs_dark_min_frames_bad_spatial``
     - 4
     - Minimum number of input frames in which a pixel must be identified as a spatial outlier before it is classified as defective.
   * - ``ifs_dark_hot_sigma``
     - 6.0
     - Number of robust standard deviations above the median dark signal used to identify hot pixels.
   * - ``ifs_dark_cold_sigma``
     - 5.0
     - Number of robust standard deviations below the median dark signal used to identify cold pixels.
   * - ``ifs_dark_temporal_sigma_thresh``
     - 10.0
     - Threshold, in robust standard deviations above the population median temporal scatter, for identifying unstable pixels. Used when no absolute cutoff is supplied.
   * - ``ifs_flat_temporal_sigma_thresh``
     - 12.0
     - Threshold, in robust standard deviations above the population median temporal scatter, for identifying unstable pixels. Used when no absolute cutoff is supplied.
   * - ``ifs_flat_spatial_sigma_thresh``
     - 8.0
     - Threshold, in units of local robust scatter, for identifying pixels whose signals deviate significantly from their surrounding pixels.
   * - ``ifs_flat_spatial_kernel_size``
     - 7
     - Size of the square median-filter kernel used to estimate the local background and spatial scatter.
   * - ``ifs_flat_min_frames_bad_spatial``
     - 3
     - Minimum number of input frames in which a pixel must be identified as a spatial outlier before it is classified as defective.
   * - ``ifs_normalize_flats``
     - False, **True**
     - Normalizes flat-field images by a locally estimated illumination pattern before identifying pixels with abnormal responses.
   * - ``ifs_flat_norm_kernel_size``
     - 31
     - Size of the median-filter kernel used to estimate the spatial illumination pattern for flat-field normalization.
   * - ``ifs_flat_low_response_thresh``
     - 0.30
     - Lower threshold for normalized flat-field response. Pixels below this value are classified as low-response (LOW_QE) pixels.
   * - ``ifs_flat_high_response_thresh``
     - 1.4
     - Upper threshold for normalized flat-field response. Pixels above this value are classified as high-response pixels.

Method use to make master calibration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``master_comb_met``
     - **median**, ivw, mean
     - Method use to combine files

Create rectification matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``create_rectmat``
     - **True**, False
     - Enables create a new rectification matrix?
   * - ``rectmat_xshift``
     - 0.0
     - Generate a second set of rectification matrices while applying a x shift to compensate for disperser motion?
   * - ``rectmat_yshift``
     - 0.0
     - Generate a second set of rectification matrices while applying a y shift to compensate for disperser motion?
   * - ``rectmat_fluxcut``
     - 0.0
     - Flux fraction threshold relative to the maximum flux, below which values are excluded.
   * - ``apply_destripe``
     - **True**, False
     - Enable additional row wise pickup noise correction.
   * - ``medres_force_ysigma``
     - **True**, False
     - A sigma cutoff in Y direction before Gaussian fit to slenslit spot in the MedRes mode.


Name of the directories
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Parameter
     - Options / selected value
     - Description
   * - ``calib_sa/plots``
     - plots
     - The directory in which the diagnostic plots are saved. Normally does not require modification.
   * - ``calib_sa/output``
     - output
     - The directory in which the data outputs are saved. Normally does not require modification.
   * - ``calib_sa/logs``
     - logs
     - The directory in which the supportive log files are saved. Normally does not require modification.
