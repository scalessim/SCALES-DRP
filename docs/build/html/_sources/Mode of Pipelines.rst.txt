Modes of Pipeline Operation
===========================

The section explains different mode of **SCALES-DRP** operation.


Quicklook Pipeline
------------------
This mode of pipeline will help the user to interact with the data during the observation.
The Quicklook Pipeline uses almost all the :doc:`Primitives <Primitives>` in a faster way.
Usually it takes a few seconds to produce the datacube using the unweighted ``Optimal extraction``.
Adding the ``Quicklook GUI`` will help the user to do it interactive and automatic after each exposure.

A user can access the ``Quicklook Pipeline`` separately using the following commands as well:

.. warning::

	Quicklook DRP can execute the data reduction of only **ONE** mode of observation at a time. So the user has to
	specify if the mode of operation is ``LOWRES``, ``MEDRES`` or 	``imaging``. A minimum number of calibration frames should be present in the directory as mentioned in :doc:`Configuration Parameters <Configuration Parameters>`.


Process files, file lists and entire directories
++++++++++++++++++++++++++++++++++++++++++++++++

- To reduce all LOWRES channel files in a directory in the order in which they
  appear and group them correctly according to the logical sequence needed by
  the pipeline:


.. code-block:: shell

   start_scales_quicklook -lr -f *.fits -g

- The corresponding MEDRES channel command would be:

.. code-block:: shell

   start_scales_quicklook -mr -f *.fits -g

Here ``-lr`` or ``-mr`` specifies the low-resolution or medium resolution mode,
``-f`` specifies file input and is followed by a file specification, and ``-g``
specifies group mode, which will group images according to what is needed by the pipeline.

* To reduce only a subset of the files (for example, a single object out of an
  entire observing run):

.. code-block:: shell

   start_scales_quicklook -lr -l input_files.lst

Here ``-l`` specifies list input mode followed by a file with a list of raw
image files for, in this case, the LOWRES mode (as indicated by the ``-lr``
parameter), one per line in the file.

* To reduce a single file:

.. code-block:: shell

   start_scales_quicklook -lr -f filename.fits


Calibration Pipeline
--------------------
The Calibration mode of pipeline will do the necessary daytime data reduction and make the
``Rectification matrix`` ready for the ``Quicklook`` and ``Science-Grad`` pipelines.
These ``Rectification matrix`` will be stored and used for :ref:`Optimal Extraction <optimalextract>`,
:ref:`Chi square extraction <leastextract>`, and the :ref:`Wavelength Calibration <wavecalib>`.

A user can access the ``Calibration Pipeline`` separately using the following commands as well:

.. warning::
  The Calibration DRP can execute the data reduction of only **ONE** mode of observation at a time. So the user has to
  specify if the mode of operation is ``LOWRES``, ``MEDRES`` or 	``imaging``. A minimum number of calibration frames
  should be present in the directory as mentioned in :doc:`Configuration Parameters <Configuration Parameters>`.

Process, file lists and entire directories
++++++++++++++++++++++++++++++++++++++++++

- To reduce all LOWRES channel files in a directory in the order in which they
  appear and group them correctly according to the logical sequence needed by
  the pipeline:


.. code-block:: shell

   start_scales_calib -lr -f *.fits -g

- The corresponding MEDRES channel command would be:

.. code-block:: shell

   start_scales_calib -mr -f *.fits -g

Here ``-lr`` or ``-mr`` specifies the low-resolution or medium resolution mode,
``-f`` specifies file input and is followed by a file specification, and ``-g``
specifies group mode, which will group images according to what is needed by the pipeline.

* To reduce only a subset of the files (for example, a single object out of an
  entire observing run):

.. code-block:: shell

   start_scales_calib -lr -l input_files.lst

Here ``-l`` specifies list input mode followed by a file with a list of raw
image files for, in this case, the LOWRES mode (as indicated by the ``-lr``
parameter), one per line in the file.


Science-Grad Pipeline
---------------------
The science-grad pipeline will do a complete data reduction explained in the remaining section of this documentation.
