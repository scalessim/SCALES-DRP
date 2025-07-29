Quick Start
===========

The ``Quick Start`` will help the users to perform the SCALES data reduction and check the installation process.

Give a quick look at the :doc:`Configuration Parameters <Configuration Parameters>` for the pipeline, contained in the file ``scalesdrp/config/scales.cfg``

The assumption is that the user have a directory ( here, data_folder) with SCLAES data (here, file_name.fits) to reduce in the ``scalesdrp`` ``conda`` environment created.
If you have the SCALES IFU data from the ``low resolution`` mode, then follow the below commands.  

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -lr -f *.fits -g


The Keck DRP Framework will load and initialize the pipeline, ingest all the ``lowres``  files, and then start processing according to the image types in groups in the order required by the pipeline. To do the same for the ``medres`` mode, data run:


.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -mr -f *.fits -g


Three directories will be created: a ``redux`` directory with the results of the reduction, a ``logs`` directory with separate logs for the framework itself and for the DRP, and a ``plots`` directory containing diagnostic plots.


For the ``Imaging`` channel, user can use the following commands.

.. code-block:: bash

   conda activate scalesdrp
   cd data_folder
   start_scales_reduce -imc -f *.fits -g
   