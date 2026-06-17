import numpy as np
from unyt import K, cm, mp, Msun
import h5py 

import sys
sys.path.append('/cosma8/data/dp004/dc-bras1/xray_absorption/LightconeIO/lightcone_io')
from lc_xray_calculator import XrayCalculator_LC

COMBINED_XRAY_EMISSIVITY_TABLE_FILENAME = "/cosma8/data/dp004/flamingo/Tables/Xray/X_Ray_table_combined.hdf5"
rng = np.random.default_rng()

def create_fake_indices(size):
    '''
    Create a latin hyper-cube of valid indices in the X-ray table
    Make the distance to these points 0 (t == 1, d == 0), in this way, the interpolation should return the array value
    Set volumes=1 and data_n=0 to avoid scaling of the interpolation result
    '''

    idx_z = rng.integers(10, size = size)
    idx_he = rng.integers(9, size = size)
    idx_T = rng.integers(45, size = size)
    idx_n = rng.integers(70, size = size)

    t_z, t_T, t_nH, t_He = np.ones(size), np.ones(size), np.ones(size), np.ones(size)
    d_z, d_T, d_nH, d_He = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)

    abundance_to_solar = np.ones((size, 11))
    joint_mask = np.ones(size, dtype = 'bool')

    volumes = np.ones(size) * cm**3
    data_n = np.zeros(size)
    
    return idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n

def sum_table_direct(band, observing_type, idx_z, idx_he, idx_T, idx_n):
    '''
    Directly take the X-ray table and sum the contributions of all metals
    This assumes solar metallicity (all metals multiplies by 1.0)
    '''
    
    tab = h5py.File(COMBINED_XRAY_EMISSIVITY_TABLE_FILENAME, 'r')
    return np.sum(10**tab[band][observing_type][()][idx_z, idx_he, :, idx_T, idx_n], axis = 1)

#initialise X-ray calculator
xray_calc = XrayCalculator_LC(
        np.array([0.0, 1.0, 2.0]), # need to interpolate over the redshift range of particles 
        COMBINED_XRAY_EMISSIVITY_TABLE_FILENAME, 
        bands=['ROSAT', 'ROSAT', 'erosita-low', 'erosita-low', 'erosita-high', 'erosita-high'], 
        observing_types=['energies_intrinsic', 'photons_intrinsic', 'energies_intrinsic', 'photons_intrinsic', 'energies_intrinsic', 'photons_intrinsic'])


#Fake indices
idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n = create_fake_indices(100000)

# Calculate luminosity from interpolation
luminosities = xray_calc.interpolate_X_Ray(
        idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n,
        bands = ['ROSAT'], observing_types = ['energies_intrinsic'], fill_value = 0)


# Directly sum the table
direct_sum = sum_table_direct('ROSAT', 'energies_intrinsic', idx_z, idx_he, idx_T, idx_n)


# assert that either the values are very small
# or the error is less than 0.1%, this is the accuracy of the interpolation from previous experiments
condition_1 = np.abs(luminosities.flatten().value - direct_sum) < 1e-42
condition_2 = np.abs(luminosities.flatten().value - direct_sum) / luminosities.flatten().value < 1e-3

assert np.sum(condition_1|condition_2) == condition_1.shape[0]
