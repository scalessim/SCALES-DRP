Installation
============

This document guide explains how to install the **SCALES-DRP** pipeline.

.. contents::
   :local:
   :depth: 2

Environment and Dependencies
----------------------------

This section explain how to create a ``conda`` environment called ‘scalesdrp’ for for SCALES pipeline and required extern dependencies.

We highly recommend using `Anaconda <https://www.anaconda.com>`_ for the installation, especially with a ``conda`` environment.

**Creating the Conda environment**

   Here we are creating a conda environment called ``scalesdrp``.

.. code-block:: bash

   conda create --name scalesdrp python=3.12
   conda activate scalesdrp

.. Note::

   If you have previously installed the DRP from source,
   we advise you to delete the ``scalesdrp`` environment and create a new one.
   To remove the conda environment, use the following command

   .. code-block:: bash

      conda remove --name scalesdrp --all


Installation Steps
------------------

This section explain the installation process directly from the source and using ``pip``.

Install from the source (`SCALES-DRP <https://github.com/scalessim/SCALES-DRP.git>`_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/scalessim/SCALES-DRP.git
   cd SCALES-DRP
   pip install .

.. Note::

   If you want to make the install editable, you can 
   follow the below command.

   .. code-block:: bash

      git clone https://github.com/scalessim/SCALES-DRP.git
      cd SCALES-DRP
      pip install -e.

Installing with ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Working on ...

.. code-block:: bash

   pip install scalesdrp

