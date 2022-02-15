## -- IMPORTS

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from astropy.io import ascii, fits
from hcipy import *


## -- FUNCTIONS 
def apply_binning(data, resolution, modes):

    # taken almost exactly from Maaike's notebook

    if resolution==modes:
        binned_data = data

    else:
        binned_data = np.zeros((modes, modes))

        binning_factor = int(resolution/modes)
        for i_index in np.arange(binning_factor, resolution+1, binning_factor):
            for j_index in np.arange(binning_factor, resolution+1, binning_factor):
                binned_col = np.mean(
                             data[i_index-binning_factor:i_index,
                                  j_index-binning_factor:j_index]
                             )
                binned_data[int((i_index)/binning_factor)-1, int((j_index)/binning_factor)-1] = binned_col

    return binned_data

def backup_apply_binning(data, resolution, modes):

    # taken almost exactly from Maaike's notebook

    if resolution==modes:
        binned_data = data

    else:
        binned_data = np.zeros((modes, modes))
        
        binning_factor = int(resolution/modes)
        print(binning_factor)
        print(resolution-binning_factor)
        print(resolution, modes)
        for i_index in np.arange(binning_factor-1, resolution-binning_factor-1, binning_factor):
            for j_index in np.arange(binning_factor-1, resolution-binning_factor-1, binning_factor):
                binned_col = np.mean(
                             data[i_index-int((binning_factor-1)/2):i_index+int((binning_factor-1)/2),
                                  j_index-int((binning_factor-1)/2):j_index+int((binning_factor-1)/2)]
                             )
                print(i_index, j_index)
                print(int((i_index+1)/binning_factor-1), int((j_index+1)/binning_factor-1))
                binned_data[int((i_index+1)/binning_factor-1), int((j_index+1)/binning_factor-1)] = binned_col

    return binned_data

def fourier_decomp(test_img, A, A_i, m, n):

    b = test_img.flatten()
    x = np.matmul(A_i, b)
    approx = np.matmul(A, x)
    if np.mean(np.abs(b - approx)) > 1e-5:
        print('STOP IT WENT WRONG')
        print(np.mean(np.abs(b - approx)))
    return x, approx, b

def velocity_to_mode(k, l, vx, vy, d=8):
    return -(k*vx + l*vy)/d

def build_complex_basis(i0, j0, m0, n0):
    A = np.zeros((m0, n0,i0*j0), dtype=complex)
    c = 0
    for i in range(i0):
        for j in range(j0):
            for m in range(m0):
                for n in range(n0):
                    A[m, n, c] = complex(np.cos(((m0-i+1)*m + (n0-j+1)*n)*2*np.pi/m0), np.sin(((m0-i+1)*m + (n0-j+1)*n)*2*np.pi/m0))
            c+=1
    A_reshape = A.reshape(m0*n0, i0*j0)

    return A_reshape


def decompose_images(i0, j0, m0, n0, resolution, t_frames, data, A, save, rcond=1e-6):
    
    modes = np.zeros((i0*j0, t_frames), dtype=complex)
    pinv = np.linalg.pinv(A, rcond=rcond)
    
    for i in range(t_frames):
        print(i)
        img = apply_binning(data[:,:, i], resolution, i0)
        coeffs, approx, mean_sub_img = fourier_decomp(img, A, pinv, m0, n0)
        modes[:, i] = coeffs
    hdu_list = fits.HDUList([fits.PrimaryHDU(np.imag(modes)), fits.ImageHDU(np.real(modes))])
    hdu_list.writeto(save, overwrite=True)
        
    return modes
    
def build_periodogram(modes, mode_sum, mode_k, mode_j, velocities, thetas, name, fs=1e3):
    
    window = signal.get_window(window='hamming', Nx=4000)
    f, Pxx_den = signal.welch(modes[mode_sum, :], fs=fs, window=window)
    sf, sP = zip(*sorted(zip(f, Pxx_den)))
    ascii.write(dict({'frequency': sf, 'signal': np.array(sP)}), f'{name}.csv', overwrite=True)
    plt.semilogy(sf, sP, label=f'Welch mode {mode_k},{mode_j}', color='cyan')
    for index, velocity in enumerate(velocities):
        layer = velocity_to_mode(mode_k, mode_j, velocity*np.cos(np.deg2rad(thetas[index])), velocity*np.sin(np.deg2rad(thetas[index])))
        plt.axvline(layer, color='gray', linestyle='--')
    plt.legend()
    plt.xlim()
    plt.ylim()
    plt.savefig(f'{plot_name}.png')


## -- Set initial conditions for sizing
# i0 : # x modes
# j0 : # y modes
# m0 : # x pixels (post binning)
# n0 : # y pixels
i0, j0, m0, n0 = 48, 48, 48, 48

## -- Read in data and create Fourier decomposition
data_file = '/data/users/jumfowle/outputs/turbulence_poyneer_8_franken_AOres=144.fits'
print(f'Reading data from {data_file}')
data = fits.getdata(data_file)

# -- Set conditions about the data we're reading in
t_frames = 60000
resolution = 144

# -- Build complex basis
print('Building complex basis.')
A = build_complex_basis(i0, j0, m0, n0)
modes = decompose_images(i0, j0, m0, n0, resolution, t_frames, data, A, save='modes=48_frames=60000_poyneer_8_franken.fits')
print('Decomposing images')

# Specify what mode and wind layers we injected
mode_sum = 300
mode_k, mode_j = 8, 16
vs = [22.7, 3.28, 16.6, 5.89, 19.8]
thetas = [246, 71, 294, 150, 14]
plot_name = 'poyneer_psd_res=144_modes=48'

print('Writing out an example.')
build_periodogram(modes, mode_sum, mode_k, mode_j, vs, thetas, plot_name)

