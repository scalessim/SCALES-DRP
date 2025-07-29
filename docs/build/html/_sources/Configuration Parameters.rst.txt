Configuration Parameters
=========================

A number of reduction parameters can be changed using entries in the configuration file.

If you installed the pipeline with ``pip``, the configuration file will not be easy to find, since it will be stored with installed ``pip`` packages. We recommend editing a copy of the config instead. You can create a copy of the config file by invoking the pipeline with the ``--write_config`` option:

.. code-block:: bash

   start_scales_reduce --write_config


.. Note ::
   
   If a parameter in the ``scales.cfg`` file is not described here, then you can assume that it should not be modified.



LOWRES and MEDRES sections of the configuration file
----------------------------------------------------

We have specific different default parameters for each observation mode. These are defined in the config file with ``[LOWRES]`` and ``[MEDRES]`` section headers. For example:

.. code-block:: bash

	bias_min_nframes = 7
	flat_min_nframes = 6
	dome_min_nframes = 3
	twiflat_min_nframes = 1
	dark_min_nframes = 3
	arc_min_nframes = 1        # = 3 for [MEDRES]
	contbars_min_nframes = 1   # = 3 for [MEDRES]
	minoscanpix = 75           # = 20 for [MEDRES]
	oscanbuf = 20              # = 5 for [MEDRES]

These parameters control the minimum number of bias, internal/dome/twilight flats and darks that the DRP expects before producing a master calibration files. The values shown here are synchronized with the calibration scripts that are used at WMKO for afternoon calibrations.



Wavelength correction parameters
--------------------------------

.. code-block:: bash

	radial_velocity_correction = "heliocentric"
	air_to_vacuum = True   # Defaults to vacuum wavelengths

These control the refinement of the wavelength solution. You can specify if you want air wavelengths by setting air_to_vacuum to False. You can specify the type of radial velocity correct as one of:
	- heliocentric
	- barycentric
	- none










