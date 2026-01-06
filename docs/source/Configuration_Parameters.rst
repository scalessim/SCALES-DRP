Configuration Parameters
=========================

A number of reduction parameters can be changed using entries in the configuration file.

If you installed the pipeline with ``pip``, the configuration file will not be easy to find, since it will be stored with installed ``pip`` packages. We recommend editing a copy of the config instead. You can create a copy of the config file by invoking the pipeline with the ``--write_config`` option:

.. code-block:: bash

   start_scales_reduce --write_config


.. Note ::
   
   If a parameter in the ``scales.cfg`` file is not described here, then you can assume that it should not be modified.



Configuration file
~~~~~~~~~~~~~~~~~~~

We have default parameters for IFS and Imaging mode of observation. For example:

.. code-block:: bash

	bias_min_nframes = 5 #minimum bias exposures
	flatlamp_min_nframes = 5 #minimum detector flat exposures
	flatlens_min_nframes = 5 #minimum lenslet flat exposures
	dark_min_nframes = 3 #minimum dark exposures


These parameters control the minimum number of bias, detector/lenslet flats and darks that the DRP expects before producing a master calibration files. The values shown here are synchronized with the calibration scripts that are used at WMKO for afternoon calibrations.












