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

import numpy as np
import scipy.interpolate
import scipy.integrate
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from tomopy.util import mproc
from tomopy_cli import log
from tomopy_cli import config

#Global variables we need for computing LUT
filters = {}
sample_material = None
scintillator_material = None
scintillator_thickness = None
ref_trans = None
d_source = 0
pixel_size = 0
center_row = 0
spectra_dict = None
possible_materials = {}

# Add a trailing slash if missing
top = os.path.join(Path(__file__).parent, '')
data_path = os.path.join(top, 'beam_hardening_data')

#Global variables for when we convert images
centerline_spline = None
angular_spline = None

class Spectrum:
    '''Class to hold the spectrum: energies and spectral power.
    '''
    def __init__(self, energies, spectral_power):
        if len(energies) != len(spectral_power):
            raise ValueError
        self.energies = energies
        self.spectral_power = spectral_power

    def fintegrated_power(self):
        return scipy.integrate.simps(self.spectral_power, self.energies)

    def fmean_energy(self):
        return scipy.integrate.simps(self.spectral_power * self.energies, self.energies) / self.fintegrated_power()
    
    def __len__(self):
        return len(energies)
 
#Copy part of the Material class from Scintillator_Optimization code
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
        return "Material({0:s}, {1:f})".format(self.name, self.density)
    
    def fread_absorption_data(self):

        raw_data = np.genfromtxt(os.path.join(data_path, self.name + '_properties_xCrossSec.dat'))
        self.energy_array = raw_data[:,0] / 1000.0      #in keV
        self.absorption_array = raw_data[:,3]   #in cm^2/g, from XCOM in XOP
        self.attenuation_array = raw_data[:,6]  #in cm^2/g, from XCOM in XOP, ignoring coherent scattering
    
    def interp_function(self,energies,absorptions):
        '''Return a function to interpolate logs of energies into logs of absorptions.
        '''
        return scipy.interpolate.interp1d(np.log(energies),np.log(absorptions),bounds_error=False)
    
    def finterpolate_absorption(self,input_energies):
        '''Interpolates absorption on log-log scale and scales back
        '''
        return np.exp(self.absorption_interpolation_function(np.log(input_energies)))
    
    def finterpolate_attenuation(self,input_energies):
        '''Interpolates attenuation on log-log scale and scales back
        '''
        return np.exp(self.attenuation_interpolation_function(np.log(input_energies)))
    
    def fcompute_proj_density(self,thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        return thickness /1e4 * self.density
    
    def fcompute_transmitted_spectrum(self,thickness,input_spectrum):
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
    
    def fcompute_absorbed_spectrum(self,thickness,input_spectrum):
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
        output_spectrum.spectral_power = (input_spectrum.spectral_power -
                - np.exp(-self.finterpolate_absorption(input_spectrum.energies) * filter_proj_density) 
                * input_spectrum.spectral_power)
        return output_spectrum
    
    def fcompute_absorbed_power(self,thickness,input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: Spectrum object for incident beam 
        Output:
        absorbed power
        '''
        return self.fcompute_absorbed_spectrum(thickness,input_spectrum).fintegrated_power()

def fread_config_file(config_filename=None):
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
                possible_materials[symbol] = Material(symbol, density)
            elif line.startswith('ref_trans'):
                global ref_trans
                ref_trans = float(line.split(':')[1].strip())


def fread_source_data():
    '''Reads the spectral power data from files.
    Data file comes from the BM spectrum module in XOP.
    Return:
    Dictionary of spectra at the various psi angles from the ring plane.
    '''
    spectra_dict = {}
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
            spectra_dict[f_angle] = Spectrum(spectral_energies, spectral_power)
    return spectra_dict


def check_material(material_str):
    '''Checks whether a material is in the list of possible materials.
    Input: string representing the symbol for a material.
    Output: Material object with the same name as the input symbol.
    '''
    for mat in possible_materials.values():
        if mat.name  == material_str:
            return mat
    else:
        raise ValueError('No such material in possible_materials: {0:s}'.format(material_str))


def add_filter(symbol, thickness):
    '''Add a filter of a given symbol and thickness.
    '''
    if symbol != 'none':
        filters[check_material(symbol)] = float(thickness)


def add_scintillator(symbol, thickness):
    '''Add a scintillator of a given symbol and thickness.
    '''
    scintillator_material = check_material(symbol)
    scintillator_thickness = float(thickness)


def parse_params(params):
    """
    Parse the input parameters to fill in filters, sample material,
    and scintillator material and thickness.
    """
    global scintillator_material
    scintillator_material = check_material(params.scintillator_material)
    global scintillator_thickness
    scintillator_thickness = params.scintillator_thickness
    add_filter(params.filter_1_material, params.filter_1_thickness)
    add_filter(params.filter_2_material, params.filter_2_thickness)
    add_filter(params.filter_3_material, params.filter_3_thickness)
    global d_source
    d_source = params.source_distance
    global sample_material
    sample_material = check_material(params.sample_material)
    global pixel_size
    pixel_size = params.pixel_size


def fapply_filters(filters, input_spectrum):
    '''Computes the spectrum after all filters.
        Inputs:
        filters: dictionary giving filter materials as keys and thicknesses in microns as values.
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        spectral power transmitted through the filter set.
        '''
    temp_spectrum = deepcopy(input_spectrum)
    for filt, thickness in filters.items():
        temp_spectrum = filt.fcompute_transmitted_spectrum(thickness, temp_spectrum)
    return temp_spectrum


def ffind_calibration_one_angle(input_spectrum):
    '''Makes a scipy interpolation function to be used to correct images.
    '''
    #Make an array of sample thicknesses
    sample_thicknesses = np.sort(np.concatenate((-np.logspace(1,0,11), [0], np.logspace(-1,4.5,221))))
    #For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                              input_spectrum)
        detected_power[i] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,
                                                                          sample_filtered_power)
    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / scintillator_material.fcompute_absorbed_power(scintillator_thickness,
                                                                                            input_spectrum)
    #Return a spline, but make sure things are sorted in ascending order
    inds = np.argsort(sample_effective_trans)
    #for i in inds:
    #    print(sample_effective_trans[i], sample_thicknesses[i])
    return InterpolatedUnivariateSpline(sample_effective_trans[inds], sample_thicknesses[inds])


def ffind_calibration(spectra_dict):
    '''Do the correlation at the reference transmission. 
    Treat the angular dependence as a correction on the thickness vs.
    transmission at angle = 0.
    '''
    angles_urad = []
    cal_curve = []
    for angle in sorted(spectra_dict.keys()):
        angles_urad.append(float(angle))
        spectrum = spectra_dict[angle]
        #Filter the beam
        filtered_spectrum = fapply_filters(filters, spectrum)
        #Create an interpolation function based on this
        angle_spline = ffind_calibration_one_angle(filtered_spectrum)
        if angle  == 0:
            global centerline_spline
            centerline_spline = angle_spline
        cal_curve.append(angle_spline(ref_trans))
    cal_curve /= cal_curve[0]
    global angular_spline
    angular_spline = InterpolatedUnivariateSpline(angles_urad, cal_curve) 


def fcorrect_as_pathlength_centerline(input_trans):
    """Corrects for the beam hardening, assuming we are in the ring plane.
    Input: transmission
    Output: sample pathlength in microns.
    """
    data_dtype = input_trans.dtype
    return_data = mproc.distribute_jobs(input_trans,centerline_spline,args=(),axis=1)
    return return_data


def fcorrect_as_pathlength(input_trans):
    '''Corrects for the angular dependence of the BM spectrum.
    First, use fconvert_data to get in terms of pathlength assuming we are
    in the ring plane.  Then, use this function to correct.
    '''
    angles = np.abs(np.arange(pathlength_image.shape[0]) - center_row)
    angles *= pixel_size / d_source
    correction_factor = angle_spline(angles)
    return centerline_spline(input_trans) * correction_factor[:,None]


def find_center_row(params):
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
    ffind_calibration(spectra_dict)
    return center_row


def initialize(params):
    '''Initializes the beam hardening correction code.
    '''
    log.info('  *** beam hardening')
    if params.beam_hardening_method != 'standard':
        log.info('   *** *** OFF')
    fread_config_file()
    global spectra_dict
    spectra_dict = fread_source_data()
    parse_params(params)
    center_row = find_center_row(params)
    log.info("  *** *** Center row for beam hardening = {0:f}".format(center_row))
    if int(params.binning) > 0:
        center_row /= pow(2, int(params.binning))
        log.info("  *** *** Center row after binning = {:f}".format(center_row))
    params.center_row = center_row
    log.info('  *** *** beam hardening initialization finished')
