# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.

from setuptools import setup, find_packages

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items("metadata"))
options = dict(conf.items("options"))

NAME = 'scalesdrp'
VERSION = '1.0.0'
RELEASE = 'dev' not in VERSION
AUTHOR = metadata["author"]
AUTHOR_EMAIL = metadata["author_email"]
LICENSE = metadata["license"]
DESCRIPTION = metadata["description"]

# scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
#            if os.path.basename(fname) != 'README.rst']
scripts = []
# Define entry points for command-line scripts
entry_points = {
    'console_scripts': [
        "start_scales_reduce = scalesdrp.scripts.reduce_scales:main",
        "start_scales_quicklook = scalesdrp.scripts.scales_quicklook:main",
        "start_scales_calib = scalesdrp.scripts.scales_calib:main"
    ]}

setup(name=NAME,
      provides=NAME,
      version=VERSION,
      license=LICENSE,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=find_packages(),
      package_data={'scalesdrp': ['configs/*.cfg', 'data/*',
                                'data/extin/*', 'data/stds/*', 'calib/*']},
      scripts=scripts,
      entry_points=entry_points,
      install_requires=['scikit-image~=0.24',
                        'astropy~=6.1.3',
                        'astroscrappy~=1.1.0',
                        'ccdproc~=2.4.2',
                        'numpy==1.26.4',
                        'scipy~=1.14.1',
                        'pyerfa',
                        'jinja2~=3.0.3',
                        'psutil~=5.7.0',
                        'pytest~=5.4.1',
                        'keckdrpframework',
                        'requests',
                        'pandas~=1.3.5',
                        'matplotlib~=3.9.2',
                        'ref_index~=1.0',
                        'pyregion~=2.0',
                        'cython',
                        'selenium',
                        'phantomjs'],
      python_requires="~=3.12"
      )
