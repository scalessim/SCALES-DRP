Configuration Parameters
=========================

A number of reduction parameters can be changed using entries in the configuration file.

If you installed the pipeline with ``pip``, the configuration file will not be easy to find, since it will be stored with installed ``pip`` packages. We recommend editing a copy of the config instead. You can create a copy of the config file by invoking the pipeline with the ``--write_config`` option:

.. code-block:: bash

   start_scales_reduce --write_config


.. Note ::
   
   If a parameter in the ``scales.cfg`` or ``scales_calib.cfg`` file is not described here, then you can assume that it should not be modified.



Configuration file
~~~~~~~~~~~~~~~~~~~

We have default parameters for IFS and Imaging mode of observation. For example:

.. code-block:: bash
	
	clobber = False #Redo the slope image generation even if the products exist
	skip_mcal_generation = True ##Skip the generation of combined master cal files & go straight to spectral extraction analysis
	rectmat_xshift = 4.0 #Generate a second set of rectification matrices while #applying an x and/or y shift to compensate for disperser motion?
	rectmat_yshift = 0


	#Calibration file names (must be inside ``SCALES-DRP/scalesdrp/calib`` directory)

	sig_map_ifs_fast0p6 = 'readnoise_ifs_fast0.6_cd5.fits' #readnoise map for IFS Fast0.6
	flat_ifs_fast0p6 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast0.6
	bpm_ifs_fast0p6 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast0.6
	bpmat_ifs_fast0p6 = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Fast0.6
	lin_coeff_ifs_fast0p6 = 'lin_coeffs_ifs_fast0.6_cd5.fits' #linearity coefficients for IFS Fast0.6

	sig_map_ifs_fast1 = 'readnoise_ifs_fast1.0_cd5.fits' #readnoise map for IFS Fast1.0
	flat_ifs_fast1 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast1.0
	bpm_ifs_fast1 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast1.0
	bpmat_ifs_fast1 = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Fast1.0
	lin_coeff_ifs_fast1 = 'lin_coeffs_ifs_fast1.0_cd5.fits' #linearity coefficients for IFS Fast1.0

	sig_map_ifs_slow = 'readnoise_ifs_slow_cd5.fits' #readnoise map for IFS Slow5.2
	flat_ifs_slow = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Slow5.2
	bpm_ifs_slow = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Slow5.2
	bpmat_ifs_slow = 'bpmat_ifs.npz' #bad pixel mask (rectification matrix) for IFS Slow5.2
	lin_coeff_ifs_slow = 'lin_coeffs_ifs_slow_cd5.fits' #linearity coefficients for IFS Slow5.2

	sig_map_img_fast0p6 = 'readnoise_img_fast0.6_cd5.fits' #readnoise map for IMG Fast0.6
	flat_img_fast0p6 = 'img_fast0.6_mflatlamp.fits' # default pseudoflat for IMG Fast0.6
	bpm_img_fast0p6 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast0.6
	bpmat_img_fast0p6 = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Fast0.6
	lin_coeff_img_fast0p6 = 'lin_coeffs_img_fast0.6_cd5.fits' #linearity coefficients for IMG Fast0.6

	sig_map_img_fast1 = 'readnoise_img_fast1.0_cd5.fits' #readnoise map for IMG Fast1.0
	flat_img_fast1 = 'img_fast1.0_mflatlamp.fits' # default pseudoflat for IMG Fast1.0
	bpm_img_fast1 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast1.0
	bpmat_img_fast1 = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Fast1.0
	lin_coeff_img_fast1 = 'lin_coeffs_img_fast1.0_cd5.fits' #linearity coefficients for IMG Fast1.0

	sig_map_img_slow = 'readnoise_img_slow_cd5.fits' #readnoise map for IMG Slow5.2
	flat_img_slow = 'img_slow_mflatlamp.fits' # default pseudoflat for IMG Slow5.2
	bpm_img_slow = 'bpm_img_cd4.fits' #bad pixel mask for IMG Slow5.2
	bpmat_img_slow = 'bpmat_img.npz' #bad pixel mask (rectification matrix) for IMG Slwo5.2
	lin_coeff_img_slow = 'lin_coeffs_img_slow_cd5.fits' #linearity coefficients for IMG Slow5.2
















