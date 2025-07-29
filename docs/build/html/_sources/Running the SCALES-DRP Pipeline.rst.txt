Running the **SCALES-DRP** Pipeline
===================================

The DRP is run using a startup script that offers several
command line options and different execution modes for each mode of analysis.

.. warning::

	DRP can exicute only **ONE** model of data reduction at a time. So the user has to
	specify if the mode of operation is ``LOWRES``, ``MEDRES`` or 	``imaging``.  


Process files, file lists and entire directories
------------------------------------------------

- To reduce all LOWRES channel files in a directory in the order in which they
  appear and group them correctly according to the logical sequence needed by
  the pipeline:


.. code-block:: shell

   start_scales_reduce -lr -f *.fits -g

- The corresponding MEDRES channel command would be:

.. code-block:: shell

   start_scales_reduce -mr -f *.fits -g

Here ``-lr`` or ``-mr`` specifies the low-resolution or medium resolution mode,
``-f`` specifies file input and is followed by a file specification, and ``-g`` 
specifies group mode, which will group images according to what is needed by the pipeline.

* To reduce only a subset of the files (for example, a single object out of an
  entire observing run):

.. code-block:: shell

   start_scales_reduce -lr -l input_files.lst

Here ``-l`` specifies list input mode followed by a file with a list of raw
image files for, in this case, the LOWRES mode (as indicated by the ``-lr``
parameter), one per line in the file.

* To reduce a single file:

.. code-block:: shell

   start_scales_reduce -lr -f filename.fits


.. Note::

	Please note that all the preliminary calibration files needed to be reduced and 
	avaialable in the data directory before reducing a science frame for all the exicution mode.  



Other command line options
--------------------------

* ``-c config_file.cfg``  This options overrides the standard configuration
  file that is stored in the installation directory in
  ``kcwidrp/config/kcwi.cfg``.

* ``--write_config`` If this option is set, an editable copy of the default DRP
  configuration file is written to wherever the command was invoked from. This
  file can then be used to modify the behavior of the pipeline using the ``-c``
  option.

* ``-k`` or ``--skipsky``  Set this to skip sky subtraction for all frames
  reduced with this command.


* ``-p proctable.proc``  When the DRP runs, it keeps track of the files
  processed using a processing table. Normally that table is called
  ``scales_lr.proc`` for the LOWRES mode and ``scales_mr.proc`` for the MEDRES mode
  and is stored in the current directory. This options is used to
  specify a different file if needed (not recommended).


