==========
tomopy-cli
==========

**tomopy** is commad-line-interface for `tomopy <https://github.com/tomopy/tomopy>`_ an open-source Python package for tomographic data processing and image reconstruction. 


Installation
============

::

    $ python setup.py install

in a prepared virtualenv or as root for system-wide installation.

.. warning:: If your python installation is in a location different from #!/usr/bin/env python please edit the first line of the bin/tomopy file to match yours.

Dependencies
============

Install the following package::

    $ conda install -c conda-forge tomopy


Usage
=====

Reconstruction
--------------

To do a tomographic reconstruction::

    $ tomopy recon --hdf-file /local/data.h5

from the command line. To get correct results, you may need to append
options such as `--center` to set the rotation axis position::

    $ tomopy recon --center 1024.0 --hdf-file /local/data.h5

to list of all available options::

    $ tomopy recon -h


Configuration File
------------------

Reconstruction parameters are stored in **tomopy.conf**. You can create a template with::

    $ tomopy init

**tomopy.conf** is constantly updated to keep track of the last stored parameters, as initalized by **init** or modified by setting a new option value. For example to re-run the last reconstrusction with identical parameters just use::

    $ tomopy recon

To run a reconstruction with a different and previously stored configuration file **old_tomopy.conf** just use::

    $ tomopy recon --config old_tomopy.conf


Find Center
-----------

To automatically find the rotation axis location of all tomographic hdf data sets in a folder (/local/data/)::

    $ tomopy find_center --hdf-file /local/data/


this generates in the /local/data/ directory a **rotation_axis.json** file containing all the automatically calculated centers::

            {"0": {"proj_0000.hdf": 1287.25}, "1": {"proj_0001.hdf": 1297.75},
            {"2": {"proj_0002.hdf": 1287.25}, "3": {"proj_0003.hdf": 1297.75},
            {"4": {"proj_0004.hdf": 1287.25}, "5": {"proj_0005.hdf": 1297.75}}

to list of all available options::

    $ tomopy find_center -h


After using **find_center**, to do a tomographic reconstruction of all tomographic hdf data sets in a folder (/local/data/)::

    $ tomopy recon --hdf-file /local/data/



