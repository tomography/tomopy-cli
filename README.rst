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
options such as `--rotation-axis` to set the rotation axis position::

    $ tomopy recon --rotation-axis 1024.0 --hdf-file /local/data.h5

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


Help
----

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
        find_center  Find rotation axis location for all hdf files in a directory

::

    $ tomopy recon -h
    usage: tomopy recon [-h] [--center-search-width CENTER_SEARCH_WIDTH]
                        [--binning {0,1,2,3}] [--blocked-views] [--dark-zero]
                        [--hdf-file PATH]
                        [--hdf-file-type {standard,flip_and_stich,mosaic}]
                        [--nsino NSINO]
                        [--nsino-per-chunk {2,4,8,16,32,64,128,256,512,1024,2048}]
                        [--reverse] [--rotation-axis ROTATION_AXIS]
                        [--rotation-axis-flip ROTATION_AXIS_FLIP]
                        [--missing-angles-end MISSING_ANGLES_END]
                        [--missing-angles-start MISSING_ANGLES_START]
                        [--zinger-level-projections ZINGER_LEVEL_PROJECTIONS]
                        [--zinger-level-white ZINGER_LEVEL_WHITE]
                        [--zinger-removal-method {none,standard}]
                        [--zinger-size ZINGER_SIZE] [--air AIR]
                        [--fix-nan-and-inf]
                        [--fix-nan-and-inf-value FIX_NAN_AND_INF_VALUE]
                        [--flat-correction-method {standard,air,none}]
                        [--minus-log]
                        [--normalization-cutoff NORMALIZATION_CUTOFF]
                        [--stripe-removal-method {none,fourier-wavelet,titarenko,smoothing-filter}]
                        [--fourier-wavelet-filter {haar,db5,sym5,sym16}]
                        [--fourier-wavelet-level FOURIER_WAVELET_LEVEL]
                        [--fourier-wavelet-pad]
                        [--fourier-wavelet-sigma FOURIER_WAVELET_SIGMA]
                        [--titarenko-alpha TITARENKO_ALPHA]
                        [--titarenko-nblock TITARENKO_NBLOCK]
                        [--smoothing-filter-size SMOOTHING_FILTER_SIZE]
                        [--alpha ALPHA] [--alpha-try] [--energy ENERGY] [--pad]
                        [--phase-retrieval-method {none,paganin}]
                        [--pixel-size PIXEL_SIZE]
                        [--propagation-distance PROPAGATION_DISTANCE]
                        [--filter {none,shepp,cosine,hann,hamming,ramlak,parzen,butterworth}]
                        [--reconstruction-algorithm {art,astrasirt,astracgls,bart,fpb,gridrec,mlem,osem,ospml_hybrid,ospml_quad,pml_hybrid,pml_quad,sirt,tv,grad,tikh}]
                        [--reconstruction-mask]
                        [--reconstruction-mask-ratio RECONSTRUCTION_MASK_RATIO]
                        [--reconstruction-type {try,slice,full}]
                        [--iteration-count ITERATION_COUNT] [--config FILE]
                        [--logs-home FILE] [--rotation-axis-file FILE] [--verbose]

    optional arguments:
      -h, --help            show this help message and exit
      --center-search-width CENTER_SEARCH_WIDTH
                            +/- center search width (pixel). Search is in 0.5
                            pixel increments (default: 10.0)
      --binning {0,1,2,3}   Reconstruction binning factor as power(2, choice)
                            (default: 0)
      --blocked-views       When set, the missing-angles options are used
                            (default: False)
      --dark-zero           When set, the the dark field is set to zero (default:
                            False)
      --hdf-file PATH       Name of the last used hdf file or directory containing
                            multiple hdf files (default: .)
      --hdf-file-type {standard,flip_and_stich,mosaic}
                            Input file type (default: standard)
      --nsino NSINO         Location of the sinogram used for slice reconstruction
                            and find axis (0 top, 1 bottom) (default: 0.5)
      --nsino-per-chunk {2,4,8,16,32,64,128,256,512,1024,2048}
                            Number of sinagram per chunk. Use larger numbers with
                            computers with larger memory (default: 32)
      --reverse             When set, the data set was collected in reverse
                            (180-0) (default: False)
      --rotation-axis ROTATION_AXIS
                            Location of rotation axis (default: 1224.0)
      --rotation-axis-flip ROTATION_AXIS_FLIP
                            Location of rotation axis in a 0-360 flip and stich
                            data collection (default: 1224.0)
      --missing-angles-end MISSING_ANGLES_END
                            Projection number of the first blocked view (default:
                            1)
      --missing-angles-start MISSING_ANGLES_START
                            Projection number of the first blocked view (default:
                            0)
      --zinger-level-projections ZINGER_LEVEL_PROJECTIONS
                            Expected difference value between outlier value and
                            the median value of the array (default: 800.0)
      --zinger-level-white ZINGER_LEVEL_WHITE
                            Expected difference value between outlier value and
                            the median value of the array (default: 1000.0)
      --zinger-removal-method {none,standard}
                            Zinger removal correction method (default: none)
      --zinger-size ZINGER_SIZE
                            Size of the median filter (default: 3)
      --air AIR             Number of pixels at each boundary to calculate the
                            scaling factor (default: 10)
      --fix-nan-and-inf     Fix nan and inf (default: False)
      --fix-nan-and-inf-value FIX_NAN_AND_INF_VALUE
                            Values to be replaced with negative values in array
                            (default: 0.0)
      --flat-correction-method {standard,air,none}
                            Flat correction method (default: standard)
      --minus-log           Minus log (default: False)
      --normalization-cutoff NORMALIZATION_CUTOFF
                            Permitted maximum vaue for the normalized data
                            (default: 1.0)
      --stripe-removal-method {none,fourier-wavelet,titarenko,smoothing-filter}
                            Stripe removal method (default: none)
      --fourier-wavelet-filter {haar,db5,sym5,sym16}
                            Type of the fourier-wavelet filter (default: sym16)
      --fourier-wavelet-level FOURIER_WAVELET_LEVEL
                            Level parameter used by the fourier-wavelet method
                            (default: 7)
      --fourier-wavelet-pad
                            When set, extend the size of the sinogram by padding
                            with zeros (default: False)
      --fourier-wavelet-sigma FOURIER_WAVELET_SIGMA
                            Damping parameter in Fourier space (default: 1)
      --titarenko-alpha TITARENKO_ALPHA
                            Damping factor (default: 1.5)
      --titarenko-nblock TITARENKO_NBLOCK
                            Number of blocks (default: 0)
      --smoothing-filter-size SMOOTHING_FILTER_SIZE
                            Size of the smoothing filter. (default: 5)
      --alpha ALPHA         Regularization parameter (default: 0.001)
      --alpha-try           When set, multiple reconstruction of the same slice
                            with different alpha coefficient are generated
                            (default: False)
      --energy ENERGY       X-ray energy [keV] (default: 20)
      --pad                 When set, extend the size of the sinogram by padding
                            with zeros (default: False)
      --phase-retrieval-method {none,paganin}
                            Phase retrieval correction method (default: none)
      --pixel-size PIXEL_SIZE
                            Pixel size [microns] (default: 1.17)
      --propagation-distance PROPAGATION_DISTANCE
                            Sample detector distance [mm] (default: 60)
      --filter {none,shepp,cosine,hann,hamming,ramlak,parzen,butterworth}
                            Reconstruction filter (default: parzen)
      --reconstruction-algorithm {art,astrasirt,astracgls,bart,fpb,gridrec,mlem,osem,ospml_hybrid,ospml_quad,pml_hybrid,pml_quad,sirt,tv,grad,tikh}
                            Reconstruction algorithm (default: gridrec)
      --reconstruction-mask
                            When set, applies circular mask to the reconstructed
                            slices (default: False)
      --reconstruction-mask-ratio RECONSTRUCTION_MASK_RATIO
                            Ratio of the mask’s diameter in pixels to the smallest
                            edge size along given axis (default: 1.0)
      --reconstruction-type {try,slice,full}
                            Reconstruct slice or full data set. For option (try):
                            multiple reconstruction of the same slice with
                            different (rotation axis) are generated (default: try)
      --iteration-count ITERATION_COUNT
                            Maximum number of iterations (default: 10)
      --config FILE         File name of configuration file (default:
                            /Users/decarlo/tomopy.conf)
      --logs-home FILE      Log file directory (default: /Users/decarlo/logs)
      --rotation-axis-file FILE
                            File name of rataion axis locations (default:
                            rotation_axis.json)
      --verbose             Verbose output (default: False)    
