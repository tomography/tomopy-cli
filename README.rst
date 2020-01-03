==========
tomopy-cli
==========

**tomopy** is commad-line-interface for `tomopy <https://github.com/tomopy/tomopy>`_ an open-source Python package for tomographic data processing and image reconstruction. 


Installation
============

Run::

    python setup.py install

in a prepared virtualenv or as root for system-wide installation.

.. warning:: If your python installation is in a location different from #!/usr/bin/env python please edit the first line of the bin/tomopy file to match yours.

Dependencies
============

Install the following packages::

    $ conda install -c conda-forge tomopy


Usage
=====

Reconstruction
--------------

To do a tomographic reconstruction::

    $ tomopy rec --hdf-file /local/data.h5

from the command line. To get correct results, you may need to append
options such as `--center` to set the rotation axis position::

    $ tomopy rec --center 1024.0 --hdf-file /local/data.h5

to list of all available options::

    $ tomopy rec -h

reconstruction parameters are stored in **tomopy.conf**. You can create a template with::

    $ tomopy init