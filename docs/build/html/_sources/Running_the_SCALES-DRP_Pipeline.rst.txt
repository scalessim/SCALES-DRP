More into the DRP Execution
=======================================

The SCALES-DRP is intergrated with  `Keck DRP framework <https://github.com/Keck-DataReductionPipelines/KeckDRPFramework>`_. The usable command lines are explained below.




Process files, file lists and entire directories
------------------------------------------------
- To reduce a single FITS file in a directory (make sure you are inside the directory where the filename.fits exist):


.. code-block:: shell

   start_scales_reduce -f filename.fits

Here ``-f`` specifies file input and is followed by a file name

- To reduce all FITS files in a directory one by one:

.. code-block:: shell

   start_scales_reduce -d FULL PATH

Here ``-d`` specifies input file directory followed by the FULL PATH to the directory.

- To reduce only a subset of the files (for example, a single object out of an
  entire observing run):

.. code-block:: shell

   start_scales_reduce -l input_files.lst

Here ``-l`` specifies list input mode followed by a file with a list of raw image files (input_files.lst)


.. Note::

	Please note that all the preliminary calibration files needed to be reduced and 
	avaialable in the data directory before reducing a science frame for all the exicution mode.  



Other command line options
--------------------------

.. csv-table:: Master files created
   :header: "Symbol", "Meaning", "Note"
   :widths: 15, 30, 30
   :file: tables/table3.csv

