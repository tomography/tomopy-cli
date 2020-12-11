=====
Usage
=====


Reconstruction
==============

To do a tomographic reconstruction::

    $ tomopy recon --file-name /local/data.h5

from the command line. To get correct results, you may need to append
options such as `--rotation-axis` to set the rotation axis position::

    $ tomopy recon --rotation-axis 1024.0 --file-name /local/data.h5

to list of all available options::

    $ tomopy recon -h


Configuration File
==================

Reconstruction parameters are stored in **tomopy.conf**. You can create a template with::

    $ tomopy init

**tomopy.conf** is constantly updated to keep track of the last stored parameters, as initalized by **init** or modified by setting a new option value. For example to re-run the last reconstrusction with identical parameters just use::

    $ tomopy recon

To run a reconstruction with a different and previously stored configuration file **old_tomopy.conf** just use::

    $ tomopy recon --config old_tomopy.conf


Output Folder
=============

The output folder for reconstructed data can be given with the
``--output-folder`` option. The other configuration parameters can be
inserted with curly braces::

  $ tomopy recon --output-folder={file_name}_rec

An additional parameter (``{file_name_parent}``) is given with the
path of the parent directory. If ``--file-name`` is a directory, then
``{file_name_parent}`` will contain the directory itself. If
``--file-name`` is a file, then ``{file_name_parent}`` will be the
parent directory of the file. The following lines will both place
reconstructed data in the directory */path/to/my/data_rec/*::

   $ tomopy recon --file-name=/path/to/my/data/file.hdf --output-folder={file_name_parent}_rec/
   $ tomopy recon --file-name=/path/to/my/data/ --output-folder={file_name_parent}_rec/


Find Center
===========

To automatically find the rotation axis location of all tomographic hdf data sets in a folder (/local/data/)::

    $ tomopy find_center --file-name /local/data/


this generates in the /local/data/ directory a **rotation_axis.json** file containing all the automatically calculated centers::

            {"0": {"proj_0000.hdf": 1287.25}, "1": {"proj_0001.hdf": 1297.75},
             "2": {"proj_0002.hdf": 1287.25}, "3": {"proj_0003.hdf": 1297.75},
             "4": {"proj_0004.hdf": 1287.25}, "5": {"proj_0005.hdf": 1297.75}}

to list of all available options::

    $ tomopy find_center -h


After using **find_center**, to do a tomographic reconstruction of all tomographic hdf data sets in a folder (/local/data/)::

    $ tomopy recon --file-name /local/data/


Manually Specifying Rotation Center
===================================

The rotation center can be specified using the ``--rotation-axis``
argument, or through a JSON file with a schema resembling::

           {"0": {"proj_0000.hdf": 1287.25}, "1": {"proj_0001.hdf": 1297.75},
            "2": {"proj_0002.hdf": 1287.25}, "3": {"proj_0003.hdf": 1297.75},
            "4": {"proj_0004.hdf": 1287.25}, "5": {"proj_0005.hdf": 1297.75}}

Including a JSON file can be beneficial if rotation centers cannot be
accurately determined automatically. The path to this JSON file can be
given as ``--rotation-axis-file`` with
``--rotation-axis-auto=json``. If ``--file-name`` points to a source
data file, only the file given by ``--file-name`` is reconstructed.
If the JSON file is also given as the argument to ``--file-name``,
then all data files listed in the JSON file will be reconstructed.

Stripe Removal
==============

Several methods of stripe removal are available in *Tomopy-cli*, and
can be selected with the ``--remove-stripe`` parameter. Each method
also has a set of associated parameters for controlling its behavior
(e.g. ``--remove-stripe=fw`` relies on ``--fw-sigma``,
``--fw-filter``, etc.).

More information about each method and the accompanying parameters can
be found in the corresponding *tomopy* documentation:

+------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Method           | ``--remove-stripe=`` Value | Tomopy Function                                                                                                                    |
+==================+============================+====================================================================================================================================+
| Fourier-Wavelet  | fw                         | `remove_stripe_fw() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_fw>`_    |
+------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Titarenko        | ti                         | `remove_stripe_ti() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_ti>`_    |
+------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Smoothing Filter | sf                         | `remove_stripe_sf() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_sf>`_    |
+------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Vo's Algorithms  | vo-all                     | `remove_all_stripe() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_all_stripe>`_  |
+------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------+

Help
====

::

    $ tomopy -h
    usage: tomopy [-h] [--config FILE] [--version]  ...
    
    optional arguments:
      -h, --help     show this help message and exit
      --config FILE  File name of configuration file
      --version      show program's version number and exit
    
    Commands:
      
        init         Create configuration file
        recon        Run tomographic reconstruction
        status       Show the tomographic reconstruction status
        segment      Run segmentation on reconstured data
        find_center  Find rotation axis location for all hdf files in a directory
        convert      Convert pre-2015 (proj, dark, white) hdf files in a single
                     data exchange h5 file
