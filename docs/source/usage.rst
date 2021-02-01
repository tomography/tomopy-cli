=====
Usage
=====


Reconstruction
==============

To do a tomographic reconstruction::

    $ tomopy recon --file-name /local/data.h5

from the command line. To get correct results, you will likely need to
provide :ref:`reconstruction parameters<reconstruction parameters>` as
options, such as ``--rotation-axis`` to set the rotation axis
position::

    $ tomopy recon --rotation-axis 1024.0 --file-name /local/data.h5

To list all available options, use::

    $ tomopy recon -h


Reconstruction Parameters
=========================

The behavior of tomopy-cli is controlled by reconstruction parameters,
which can be given in one of three ways, in order of precedence:

1. Directly as :ref:`command line arguments` (e.g. ``--rotation-axis=1024.0``).
2. Per-tomogram in a :ref:`YAML parameter file` given as the argument to the ``--parameter-file`` option.
3. In the :ref:`global configuration file`.


Command Line Arguments
----------------------

The **simplest way** to give a reconstruction parameter is to directly
pass it as an option to the ``tomopy`` command. Some options also
accept an argument, while others simple enable certain
behavior. Parameters given directly via the command line will override
those given via a parameter file or global configuration file.

For example, to specify the rotation center of 1023, apply Vo's stripe
removal algorithms, and enable automatic reading of the pixel size for
a tomogram stored in *my_tomogram.h5*, use::

    $ tomopy recon --file-name my_tomogram.h5 --rotation-center 1023.0 --remove-stripe-method vo-all --pixel-size-auto


YAML Parameter File
-------------------

In some cases, many tomograms need to be reconstructed with slightly
different parameters. A common example of this is the rotation axis
(discussed in detail :ref:`below<find center>`), which may drift over
the course of an operando experiment. This can be done with a YAML
file containing the extra parameters for each tomogram::

  my_tomogram_001.h5:
    rotation-axis: 1023.0
  my_tomogram_002.h5:
    rotation-axis: 1025.0
    reconstruction-algorithm: gridrec

Any parameters specified in this way will override those in the
:ref:`global configuration file`. The filenames listed in the YAML
file can be relative to the current working directory, including
subdirectories, but cannot use other file-system shortcuts (e.g. “..”,
“~”).

.. warning::

   *tomopy-cli* does not modify parameters other than those given. In
   the above example, specifying a rotation axis will have no effect
   if ``--rotation-axis-auto=auto``, since the rotation axis will be
   calculated, and the value of ``--rotation-axis`` is then ignored.


Global Configuration File
-------------------------

Reconstruction parameters can also be stored in a global configuration
file specified with the ``--config`` option (default:
*tomopy.conf*). You can create a template with::

    $ tomopy init

If *tomopy-cli* is invoked with the ``--config-update`` option, then
the configuration file is updated to keep track of the last stored
parameters, as initalized by ``$ tomopy init`` or modified by setting
a new option value. For example to re-run the last reconstrusction
with identical parameters just use::

    $ tomopy recon

To run a reconstruction with a different and previously stored
configuration file *alternate_tomopy.conf* just use::

    $ tomopy recon --config alternate_tomopy.conf


Output Folder
=============

The output folder for reconstructed data can be given with the
``--output-folder`` option. The other configuration parameters can be
inserted with curly braces::

  $ tomopy recon --output-folder={file_name}_rec

An additional parameter (``{file_name_parent}``) is available with the
path of the parent directory. If ``--file-name`` is a directory, then
``{file_name_parent}`` will contain the directory itself. If
``--file-name`` is a file, then ``{file_name_parent}`` will be the
parent directory of the file. The following lines will both place
reconstructed data in the directory */path/to/my/data_rec/*::

   $ tomopy recon --file-name=/path/to/my/data/file.hdf --output-folder={file_name_parent}_rec/
   $ tomopy recon --file-name=/path/to/my/data/ --output-folder={file_name_parent}_rec/


Find Center
===========

To automatically find the rotation axes locations of all tomographic
HDF data sets in a folder (e.g. */local/data/*), use::

    $ tomopy find_center --file-name /local/data/


this generates, in the */local/data/* directory, a YAML file (default:
*extra_params.yaml*) containing all the automatically calculated
centers::

    proj_0000.hdf:
        rotation-axis: 1287.25
    proj_0001.hdf:
        rotation-axis: 1297.75
    proj_0002.hdf:
        rotation-axis: 1287.25
    proj_0003.hdf:
        rotation-axis: 1297.75
    proj_0004.hdf:
        rotation-axis: 1287.25
    proj_0005.hdf:
        rotation-axis: 1297.75

If the YAML file already exists, it will be updated with the new
rotation axes.

To list all available options::

    $ tomopy find_center -h

After using ``$ tomopy find_center``, one can do tomographic
reconstructions of all tomographic HDF data sets in a folder
(e.g. */local/data/*) with::

    $ tomopy recon --file-name /local/data/


Stripe Removal
==============

Several methods of stripe removal are available in *tomopy-cli*, and
can be selected with the ``--remove-stripe-method`` parameter. Each
method may also have a set of associated parameters for controlling
its behavior (e.g. ``--remove-stripe-method=fw`` relies on
``--fw-sigma``, ``--fw-filter``, etc.).

More information about each method and the accompanying parameters can
be found in the corresponding *tomopy* documentation:

+------------------+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Method           | ``--remove-stripe-method=`` Value | Tomopy Function                                                                                                                    |
+==================+===================================+====================================================================================================================================+
| Fourier-Wavelet  | fw                                | `remove_stripe_fw() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_fw>`_    |
+------------------+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Titarenko        | ti                                | `remove_stripe_ti() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_ti>`_    |
+------------------+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Smoothing Filter | sf                                | `remove_stripe_sf() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_stripe_sf>`_    |
+------------------+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Vo's Algorithms  | vo-all                            | `remove_all_stripe() <https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html#tomopy.prep.stripe.remove_all_stripe>`_  |
+------------------+-----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+

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
