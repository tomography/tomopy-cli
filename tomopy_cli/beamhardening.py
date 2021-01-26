'''Code to correct for beam hardening effects in tomography experiments.
The main application of this code is for synchrotron experiments with 
a bending magnet beam.  This beam is both polychromatic and has a spectrum
which varies with the vertical angle from the ring plane.  In principle,
this could be used for other polychromatic x-ray sources.

The mathematical approach is to filter the incident spectrum by a 
series of filters.  This filtered spectrum passes through a series of
thicknesses of the sample material.  For each thickness, the transmitted
spectrum illuminates the scintillator material.  The absorbed power in 
the scintillator material is computed as a function of the
sample thickness.  A univariate spline fit is then calculated
between the calculated transmission and the sample thickness for the centerline
of the BM fan.  This is then used as a lookup table to convert sample 
transmission to sample thickness, as an alternative to Beer's law.
To correct for the dependence of the spectrum on vertical angle,
at a reference transmission (0.1 by default, which works well with the APS BM
beam), the ratio between sample thickness computed with the centerline spline
fit and the actual sample thickness is computed as a correction factor. 
A second spline fit between vertical angle and correction factor is calculated,
and this is used to correct the images for the dependence of the spectrum
on the vertical angle in the fan.  

This code uses a set of text data files to both define the spectral
properties of the beam and to define the absorption and attenuation
properties of various materials.  

The spectra are in text files with 
two columns.  The first column gives energy in eV, the second the spectral
power of the beam.  A series of files are used, in the form 
Psi_##urad.dat, with the ## corresponding to the vertical angle from the ring
plane in microradians.  These files were created in the BM spectrum
tool of XOP.

The spectral properties of the various filter, sample, and scintillator 
materials were computed in XOP with the xCrossSec tool.  To add a new
material, compute the spectral properties with xCrossSec and add the 
file to the beam_hardening_data folder.

This code also uses a setup.cfg file, located in beam_hardening_data.
This mainly gives the options for materials, their densities, and 
the reference transmission for the angular correction factor.

Usage:
* Run fread_config_file to load in configuration information.
* Run fcompute_calibrations to compute the polynomial fits for correcting
    the beam hardening
* Run either fcorrect_as_pathlength or fcorrect_as_transmission, as desired,
    to correct an image.

'''
from copy import deepcopy
import os
from pathlib import Path, PurePath
import logging
from typing import Mapping

import numpy as np
import scipy.interpolate
import scipy.integrate
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from tomopy.util import mproc
from tomopy_cli import config


log = logging.getLogger(__name__)


data_path = Path(__file__).parent / 'beam_hardening_data'


class Spectrum:
    '''Class to hold the spectrum: energies and spectral power.'''
    def __init__(self, energies, spectral_power):
        if len(energies) != len(spectral_power):
            raise ValueError
        self.energies = energies
        self.spectral_power = spectral_power

    def fintegrated_power(self):
        return scipy.integrate.simps(self.spectral_power, self.energies)

    def fmean_energy(self):
        power = self.spectral_power
        total_power = self.fintegrated_power()
        energies = self.energies
        return scipy.integrate.simps(power * energies, energies) / total_power
    
    def __len__(self):
        return len(energies)


# Copy part of the Material class from Scintillator_Optimization code
class Material:
    '''Class that defines the absorption and attenuation properties of a material.
    
    Data based off of the xCrossSec database in XOP 2.4.
    
    '''
    def __init__(self,name,density):
        self.name = name
        self.density = density  #in g/cc
        self.fread_absorption_data()
        self.absorption_interpolation_function = self.interp_function(self.energy_array,self.absorption_array)
        self.attenuation_interpolation_function = self.interp_function(self.energy_array,self.attenuation_array)
    
    def __repr__(self):
        return "Material({0:s}, {1:f}g/mL)".format(self.name, self.density)
    
    def fread_absorption_data(self):
        raw_data = np.genfromtxt(os.path.join(data_path, self.name + '_properties_xCrossSec.dat'))
        self.energy_array = raw_data[:,0] / 1000.0      #in keV
        self.absorption_array = raw_data[:,3]   #in cm^2/g, from XCOM in XOP
        self.attenuation_array = raw_data[:,7]  #in cm^2/g, from XCOM in XOP, ignoring coherent scattering
    
    def interp_function(self,energies,absorptions):
        '''Return a function to interpolate logs of energies into logs of absorptions.
        '''
        return scipy.interpolate.interp1d(np.log(energies), np.log(absorptions), bounds_error=False)
    
    def finterpolate_absorption(self, input_energies):
        '''Interpolates absorption on log-log scale and scales back
        '''
        return np.exp(self.absorption_interpolation_function(np.log(input_energies)))
    
    def finterpolate_attenuation(self,input_energies):
        '''Interpolates attenuation on log-log scale and scales back
        '''
        return np.exp(self.attenuation_interpolation_function(np.log(input_energies)))
    
    def fcompute_proj_density(self, thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        return thickness /1e4 * self.density
    
    def fcompute_transmitted_spectrum(self, thickness, input_spectrum):
        '''Computes the transmitted spectral power through a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident spectrum
        Output:
        Spectrum object for transmitted intensity
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        output_spectrum.spectral_power *= (
                    np.exp(-self.finterpolate_attenuation(output_spectrum.energies) * filter_proj_density))
        return output_spectrum
    
    def fcompute_absorbed_spectrum(self, thickness, input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident beam
        Output:
        Spectrum objection for absorbed spectrum
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        ext_lengths = self.finterpolate_absorption(input_spectrum.energies) * filter_proj_density 
        output_spectrum.spectral_power *= (1.0 - np.exp(-ext_lengths))
        return output_spectrum
    
    def fcompute_absorbed_power(self, thickness, input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: Spectrum object for incident beam 
        Output:
        absorbed power
        '''
        return self.fcompute_absorbed_spectrum(thickness,input_spectrum).fintegrated_power()


def fapply_filters(filters: Mapping, input_spectrum):
    """Computes the spectrum after all filters.
    
    Parameters
    ==========
    filters
      Holds the loaded filters described by ``filters[symbol] =
      thickness``.
    input_spectrum : np.ndarray
      spectral power for input spectrum, in numpy array
    
    Returns
    =======
    temp_spectrum : np.ndarray
      spectral power transmitted through the filter set.
    
    """
    temp_spectrum = deepcopy(input_spectrum)
    for filt, thickness in filters.items():
        temp_spectrum = filt.fcompute_transmitted_spectrum(thickness, temp_spectrum)
    return temp_spectrum


class BeamSoftener():
    # Variables we need for computing LUT
    spectra_dict = None # Initialized in __init__
    possible_materials = None # Initialized in __init__
    scintillator_thickness = 0
    scintillator_material = None
    d_source = None
    sample_material = None
    pixel_size = None
    filters = None # Initialized in __init__
    angular_spline = None
    ref_trans = None
    threshold_trans = None
    # Variables for when we convert images
    centerline_spline = None
    angular_spline = None

    
    def __init__(self, params):
        """Initializes the beam hardening correction code."""
        log.info('  *** beam hardening')
        self.possible_materials = {}
        self.filters = {}        
        if params.beam_hardening_method == 'standard':
            self.fread_config_file()
            self.fread_source_data()
            self.parse_params(params)
            self.center_row = self.find_center_row(params)
            log.info("  *** *** Center row for beam hardening = {0:f}".format(self.center_row))
            if int(params.binning) > 0:
                self.center_row /= pow(2, int(params.binning))
                log.info("  *** *** Center row after binning = {:f}".format(self.center_row))
            params.center_row = self.center_row
            log.info('  *** *** beam hardening initialization finished')
        else:
            log.info('   *** *** OFF')
    
    def fread_config_file(self, config_filename=None):
        '''Read in parameters for beam hardening corrections from file.
        Default file is in same directory as this source code.
        Users can input an alternative config file as needed.
        '''
        if config_filename:
            config_path = Path(config_filename)
            if not config_path.exists():
                raise IOError('Config file does not exist: ' + str(config_path))
        else:
            config_path = Path.joinpath(Path(__file__).parent, 'beam_hardening_data', 'setup.cfg')
        with open(config_path, 'r') as config_file:
            while True:
                line = config_file.readline()
                if line == '':
                    break
                if line.startswith('#'):
                    continue
                elif line.startswith('symbol'):
                    symbol = line.split(',')[0].split('=')[1].strip()
                    density = float(line.split(',')[1].split('=')[1])
                    self.possible_materials[symbol] = Material(symbol, density)
                elif line.startswith('ref_trans'):
                    self.ref_trans = float(line.split(':')[1].strip())
                elif line.startswith('threshold_trans'):
                    self.threshold_trans = float(line.split(':')[1].strip())
    
    def fread_source_data(self):
        """Reads the spectral power data from files.  Data file comes from the
        BM spectrum module in XOP. Saves *self.spectra_dict*: a
        dictionary of spectra at the various psi angles from the ring
        plane.
        
        """
        self.spectra_dict = {}
        file_list = list(filter(lambda x: x.endswith(('.dat', '.DAT')), os.listdir(data_path)))
        for f_name in file_list:
            f_path = os.path.join(data_path, f_name)
            #print(f_path)
            if os.path.isfile(f_path) and f_name.startswith('Psi'):
                log.info('  *** *** source file {:s} located'.format(f_name))
                f_angle = float(f_name.split('_')[1][:2])
                spectral_data = np.genfromtxt(f_path, comments='!')
                spectral_energies = spectral_data[:,0] / 1000.
                spectral_power = spectral_data[:,1]
                self.spectra_dict[f_angle] = Spectrum(spectral_energies, spectral_power)
    
    def parse_params(self, params):
        """Parse the input parameters to fill in filters, sample material,
        and scintillator material and thickness.
        
        """
        self.scintillator_material = self.check_material(params.scintillator_material)
        self.scintillator_thickness = params.scintillator_thickness
        self.add_filter(params.filter_1_material, params.filter_1_thickness)
        self.add_filter(params.filter_2_material, params.filter_2_thickness)
        self.add_filter(params.filter_3_material, params.filter_3_thickness)
        self.d_source = params.source_distance
        self.sample_material = self.check_material(params.sample_material)
        log.info("  *** *** Sample material: %s", self.sample_material)
        self.pixel_size = params.pixel_size
    
    def check_material(self, material: str):
        """Checks whether a material is in the list of possible materials.
        
        Parameters
        ==========
        material:
          Symbol for a material.
        
        Returns
        =======
        mat
          Material object with the same name as the input symbol.
        
        Raises
        ======
        ValueError
          Material symbol is unknown.
        
        """
        for mat in self.possible_materials.values():
            if mat.name  == material:
                return mat
        else:
            raise ValueError('No such material in possible_materials: {0:s}'.format(material))
    
    def add_filter(self, symbol, thickness):
        """Add a filter of a given symbol and thickness."""
        if symbol != 'none':
            self.filters[self.check_material(symbol)] = float(thickness)
    
    def find_center_row(self, params):
        '''Finds the brightest row of the input image.
        Filters to make sure we ignore spurious noise.
        '''
        with h5py.File(params.file_name,'r') as hdf_file:
            bright = hdf_file['/exchange/data_white'][...]
        if len(bright.shape) > 2:
            bright = bright[-1,...]
        vertical_slice = np.sum(bright, axis=1, dtype=np.float64)
        gaussian_filter = scipy.signal.windows.gaussian(200,20)
        filtered_slice = scipy.signal.convolve(vertical_slice, gaussian_filter,
                                                mode='same')
        center_row = float(np.argmax(filtered_slice))
        self.ffind_calibration()
        return center_row
    
    def ffind_calibration(self):
        """Do the correlation at the reference transmission.  Treat the
        angular dependence as a correction on the thickness vs.
        transmission at angle = 0.
        
        """
        angles_urad = []
        cal_curve = []
        for angle in sorted(self.spectra_dict.keys()):
            angles_urad.append(float(angle))
            spectrum = self.spectra_dict[angle]
            #Filter the beam
            filtered_spectrum = fapply_filters(self.filters, spectrum)
            #Create an interpolation function based on this
            angle_spline = self.ffind_calibration_one_angle(filtered_spectrum)
            if angle  == 0:
                self.centerline_spline = angle_spline
            cal_curve.append(angle_spline(self.ref_trans))
        cal_curve /= cal_curve[0]
        self.angular_spline = InterpolatedUnivariateSpline(angles_urad, cal_curve) 
    
    def ffind_calibration_one_angle(self, input_spectrum):
        '''Makes a scipy interpolation function to be used to correct images.
        
        '''
        # Make an array of sample thicknesses
        sample_thicknesses = np.sort(np.concatenate((-np.logspace(1,0,21), [0], np.logspace(-1,4.5,441))))
        # For each thickness, compute the absorbed power in the scintillator
        detected_power = np.zeros_like(sample_thicknesses)
        for i in range(sample_thicknesses.size):
            sample_filtered_power = self.sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                                  input_spectrum)
            detected_power[i] = self.scintillator_material.fcompute_absorbed_power(self.scintillator_thickness,
                                                                                   sample_filtered_power)
        # Compute an effective transmission vs. thickness
        absorbed_power = self.scintillator_material.fcompute_absorbed_power(self.scintillator_thickness,
                                                                            input_spectrum)
        sample_effective_trans = detected_power / absorbed_power
        # Threshold the transmission we accept to keep the spline from getting unstable
        usable_trans = sample_effective_trans[sample_effective_trans > self.threshold_trans]
        usable_thicknesses = sample_thicknesses[sample_effective_trans > self.threshold_trans]
        # Return a spline, but make sure things are sorted in ascending order
        inds = np.argsort(usable_trans)
        return InterpolatedUnivariateSpline(usable_trans[inds], usable_thicknesses[inds], ext='const')

    def fcorrect_as_pathlength_centerline(self, input_trans):
        """Corrects for the beam hardening, assuming we are in the ring plane.

        Parameters
        ==========
        input_trans : np.ndarray
          transmission

        Returns
        =======
        pathlength : np.ndarray
          sample pathlength in microns.

        """
        data_dtype = input_trans.dtype
        pathlength = mproc.distribute_jobs(input_trans, self.centerline_spline, args=(), axis=1)
        return pathlength

    def fcorrect_as_pathlength(self, input_trans):
        '''Corrects for the angular dependence of the BM spectrum.
        First, use fconvert_data to get in terms of pathlength assuming we are
        in the ring plane.  Then, use this function to correct.
        '''
        angles = np.abs(np.arange(input_trans.shape[0]) - self.center_row)
        angles *= self.pixel_size / self.d_source
        correction_factor = self.angular_spline(angles)
        return self.centerline_spline(input_trans) * correction_factor[:,None]
    
