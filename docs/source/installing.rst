Installation
============

This document guide explains how to install the **SCALES-DRP** pipeline.

.. contents::
   :local:
   :depth: 2

Requirements
------------

Before installing, ensure you have the following installed:

- Python 3.12
- pip
- Git
- (Optional) Anaconda or virtualenv for managing environments

Dependencies
------------

The required Python packages can be installed from a `requirements.txt` file.

Installation Steps
------------------

**1. Clone the Repository**

.. code-block:: bash

   git clone https://github.com/scalessim/SCALES-DRP.git
   cd SCALES-DRP

**2. (Optional) Create a Virtual Environment**

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate

**3. Install Dependencies**

.. code-block:: bash

   pip install -r requirements.txt

**4. Build the Documentation (Optional)**

.. code-block:: bash

   cd docs
   make html

**5. Run a Test (Optional)**

To verify the installation:

.. code-block:: bash

   python -m scalessim  # Or replace with actual test command

Advanced Installation
---------------------

If you need to install system-level libraries or external dependencies:

.. code-block:: bash

   sudo apt install gcc gfortran
   # or for MacOS:
   brew install gcc

Troubleshooting
---------------

- If you encounter an error like `ModuleNotFoundError`, ensure all dependencies are installed.
- Use `pip list` to confirm that required packages like `sphinx`, `numpy`, or `scipy` are available.

---

Let me know if you'd like this customized for **molecfit**, **your tel_pipeline**, or **SCALES-DRP** specifically—I can help tailor the instructions!