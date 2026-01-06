:orphan:

Quick Start
===========

The ``Quick Start`` will help the users to perform the SCALES data reduction and check the installation process.

Give a quick start at the :doc:`Configuration Parameters <Configuration_Parameters>` for the pipeline, contained in the file ``scalesdrp/config/scales.cfg``

The assumption is that the user have a directory ( here, data_folder) with SCLAES data (here, file_name.fits) to reduce in the ``scalesdrp`` ``conda`` environment created.
If you have the SCALES IFU data from the ``low resolution`` mode, then follow the below commands.  

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -lr -f file_name.fits


The Keck DRP Framework will load and initialize the pipeline, ingest the input ``lowres``  file, and then start processing according to the input image type. To do the same for the ``medres`` mode, data run:


.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -mr -f file_name*.fits


Three directories will be created: a ``redux`` directory with the results of the reduction, a ``logs`` directory with separate logs for the framework itself and for the DRP, and a ``plots`` directory containing diagnostic plots.


For the ``Imaging`` channel, user can use the following commands.

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -imc -f file_name.fits
   
.. note::

   It is important to execute the above commands from the data directory. One can also specify
   the complete path following ``-d``. For example:

   .. code-block:: bash

      start_scales_reduce -imc -f -d /FULL_PATH file_name.fits
