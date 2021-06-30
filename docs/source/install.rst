=======
Install
=======

Installation from Source
========================

::

    $ git clone https://github.com/tomography/tomopy-cli.git
    $ cd tomopy-cli
    $ python setup.py install

in a prepared virtualenv or as root for system-wide installation.

.. warning:: If your python installation is in a location different from #!/usr/bin/env python please edit the first line of the bin/tomopy file to match yours.

After the installation you will be prompted to::

    $ source /Users/userid/complete_tomopy.sh

This enable the autocompletition of all tomopy recon options. Just press tab after::

    $ tomopy recon --<TAB> <TAB>
    
to select an optional parameter and show its default value.
 
.. warning:: in some systems you are required to set *complete_tomopy.sh* as executable with::

    $ chmod +x /Users/userid/complete_tomopy.sh

Update
======

**tomopy-cli** is constantly updated to include new features. To update your locally installed version::

    $ cd tomopy-cli
    $ git pull
    $ python setup.py install


Dependencies
============

Install the following package::

    $ conda install -c conda-forge tomopy

Optionally, *dxchange* may be needed for file I/O::

    $ conda install -c conda-forge dxchange

