Configuration Parameters
=========================

A number of reduction parameters can be changed using entries in the configuration file.

If you installed the pipeline with ``pip``, the configuration file will not be easy to find,
since it will be stored with installed ``pip`` packages. We recommend editing a copy of the
config instead. You can create a copy of the config file by invoking the pipeline with
the ``--write_config`` option:

.. code-block:: bash

   start_scales_reduce --write_config


.. Note ::

   If a parameter in the ``scales.cfg`` or ``scales_calib.cfg`` file is not described here,
   then you can assume that it should not be modified.

Configuration Parameters Common to both the module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have default parameters for both calibration module and science-grade module operations.
The configuration parameters corresponding to calibration and science-grade module
can be found in  ``scales_calib.cfg`` and ``scales.cfg``, respectively.
The list of common tunable parameters are listed below.

.. code-block:: bash
	clobber = False #Redo the slope image generation even if the products exist
  output_directory = 'redux/' #output directory name

Dectector level corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

  do_swap = False #apply odd even column swapping on the raw reads.
  nchans=4 #Number of readout channel on both the detectors
  altcol=True #ACN correction using odd and even column separately for each channel
  channelwise=True #ACN correction using a single value per channel
  amp_mean_func=robust.mean #ACN correction function
  do_acn=True # ACN 2nd level correction needed?
  acn_avg_type='pix' #'pix' or 'frame'
  acn_mean_func=np.median # ACN 2nd level correction function
  acn_smooth=True #smoothening needed?
  acn_savgol=False #savgol filter for smoothening
  acn_winsize=31 #smoothening window size
  acn_order=3
  resid_colsub=False #enable residual column subtraction
  fixcol=True #enable 1/f correction
  ref_avg_type='row_wise'#'frame', 'pix', 'row_wise'
  ref_mean_func=np.median
  ref_smooth=True
  ref_savgol=False
  ref_winsize=31
  ref_order=3
  pickup=False #enable pickup noise correction
  sigma_thresh=4.0
  dilate_iter=2
  highpass_size=101
  per_amp=False

  do_linearity = True # Do you wanted to apply linearity correction?
  apply_sat_mask=True #Do you like to apply saturation mask to ramp fitting
  apply_bpm = True #Do you like to apply bad pixel mask to the slope image?

  apply_dark=True #Do you like to apply master dark correction if the file exist?
  apply_det_flat=True #Do you like to apply master detector flat correction if the file exist?
  apply_bias=True Do you like to apply master bias correction if the file exist?

Calibration Files associated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is the name of calibration files required for the data analysis.
All the calibration files should be added to ``SCALES-DRP/scalesdrp/calib`` directory.

.. code-block:: bash

  sig_map_ifs_fast0p6 = 'readnoise_ifs_fast0.6_cd5.fits' #readnoise map for IFS Fast0.6
  flat_ifs_fast0p6 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast0.6
  bpm_ifs_fast0p6 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast0.6
  bpmat_ifs_fast0p6 = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Fast0.6
  lin_coeff_ifs_fast0p6 = 'lin_coeffs_ifs_fast0.6_cd5.fits' #linearity coefficients for IFS Fast0.6
  sat_map_ifs_fast0p6 = 'ifs_fast0.6_sat_map.fits'

  sig_map_ifs_fast1 = 'readnoise_ifs_fast1.0_cd5.fits' #readnoise map for IFS Fast1.0
  flat_ifs_fast1 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast1.0
  bpm_ifs_fast1 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast1.0
  bpmat_ifs_fast1 = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Fast1.0
  lin_coeff_ifs_fast1 = 'lin_coeffs_ifs_fast1.0_cd5.fits' #linearity coefficients for IFS Fast1.0
  sat_map_ifs_fast1 = 'ifs_fast1.0_sat_map.fits'

  sig_map_ifs_slow = 'readnoise_ifs_slow_cd5.fits' #readnoise map for IFS Slow5.2
  flat_ifs_slow = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Slow5.2
  bpm_ifs_slow = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Slow5.2
  bpmat_ifs_slow = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Slow5.2
  lin_coeff_ifs_slow = 'lin_coeffs_ifs_slow_cd5.fits' #linearity coefficients for IFS Slow5.2
  sat_map_ifs_slow = 'ifs_slow_sat_map.fits'

  sig_map_img_fast0p6 = 'readnoise_img_fast0.6_cd5.fits' #readnoise map for IMG Fast0.6
  flat_img_fast0p6 = 'img_fast0.6_mflatlamp.fits' # default pseudoflat for IMG Fast0.6
  bpm_img_fast0p6 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast0.6
  bpmat_img_fast0p6 = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Fast0.6
  lin_coeff_img_fast0p6 = 'lin_coeffs_img_fast0.6_cd5.fits' #linearity coefficients for IMG Fast0.6
  sat_map_img_fast0p6 = 'img_fast0.6_sat_map.fits'

  sig_map_img_fast1 = 'readnoise_img_fast1.0_cd5.fits' #readnoise map for IMG Fast1.0
  flat_img_fast1 = 'img_fast1.0_mflatlamp.fits' # default pseudoflat for IMG Fast1.0
  bpm_img_fast1 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast1.0
  bpmat_img_fast1 = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Fast1.0
  lin_coeff_img_fast1 = 'lin_coeffs_img_fast1.0_cd5.fits' #linearity coefficients for IMG Fast1.0
  sat_map_img_fast1 = 'img_fast1.0_sat_map.fits'

  sig_map_img_slow = 'readnoise_img_slow_cd5.fits' #readnoise map for IMG Slow5.2
  flat_img_slow = 'img_slow_mflatlamp.fits' # default pseudoflat for IMG Slow5.2
  bpm_img_slow = 'bpm_img_cd4.fits' #bad pixel mask for IMG Slow5.2
  bpmat_img_slow = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Slwo5.2
  lin_coeff_img_slow = 'lin_coeffs_img_slow_cd5.fits' #linearity coefficients for IMG Slow5.2
  sat_map_img_slow = 'img_slow_sat_map.fits'

Calibration Module Specific Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have configuration parameters specific to calibration module. They are listed below:

.. code-block:: bash

	skip_mcal_generation = True ##Skip the generation of combined master cal files & go straight to spectral extraction analysis
	rectmat_xshift = 4.0 #Generate a second set of rectification matrices while #applying an x and/or y shift to compensate for disperser motion?
	rectmat_yshift = 0


Science-grade Module Specific Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have default parameters for science-grade module listed below.
The name of the default rectification matrix associated with each IFS module.
All these calibration files should be present in the ``SCALES-DRP/scalesdrp/calib`` directory.

.. code-block:: bash

  OPT_rmat_LowRes_K = 'LowRes-K_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_K = 'LowRes-K_C2_intp_rectmat.npz'
  #OPT_rmat_LowRes_L = 'LowRes-L_OPT_intp_rectmat.npz'
  #C2_rmat_LowRes_L = 'LowRes-L_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_L = 'LowRes-L_OPT_rectmat.npz'
  C2_rmat_LowRes_L = 'LowRes-L_C2_rectmat.npz'
  OPT_rmat_LowRes_M = 'LowRes-M_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_M = 'LowRes-M_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_KLM = 'LowRes-KLM_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_KLM = 'LowRes-KLM_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_KL = 'LowRes-KL_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_KL = 'LowRes-KL_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_Ls = 'LowRes-Ls_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_Ls = 'LowRes-Ls_C2_intp_rectmat.npz'
  OPT_rmat_MedRes_K = 'MedRes-K_OPT_intp_rectmat_260604.npz'
  #OPT_rmat_MedRes_K = 'MedRes-K_OPT_intp_rectmat_dx4.0_dy0.npz'
  C2_rmat_MedRes_K = 'MedRes-K_C2_intp_rectmat_260604.npz'
  #C2_rmat_MedRes_K = 'MedRes-K_C2_intp_rectmat_dx4.0_dy0.npz'
  OPT_rmat_MedRes_L = 'MedRes-L_OPT_intp_rectmat.npz'
  C2_rmat_MedRes_L = 'MedRes-L_C2_intp_rectmat.npz'
  OPT_rmat_MedRes_M = 'MedRes-M_OPT_intp_rectmat.npz'
  C2_rmat_MedRes_M = 'MedRes-M_C2_intp_rectmat.npz'

  do_chi2_full = False #enable chi square spectral extraction
  chi2_method_lowres = 'lsmr' # per-lenslet chi2 solver for LowRes: 'lsmr' (iterative) or 'dense' (direct SVD)
  chi2_method_medres = 'lsmr' # per-lenslet chi2 solver for MedRes
  chi2_n_workers = 'auto' # worker processes for the chi2 solve; 'auto' = CPUs - 1, 1 = serial
  chi2_block_cache = 'disk' # cache the rectmat's per-lenslet blocks: 'disk', 'memory' or 'off'
  chi2_block_cache_dir = '' # disk cache location; empty = <output_directory>/cache
  apply_lens_flat=False # apply lenslet flat correction
  lowres_final_cube = (54, 112, 112) # final cube shape of the lowRes IFS mode
  medres_final_cube = (1900, 17, 18) # final cube shape of the MedRes IFS mode
