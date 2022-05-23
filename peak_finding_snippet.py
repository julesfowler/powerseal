## - snippet / spaghetti code for peak ID

from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal



def perfect_peak(freq, sigma, alpha):
    omega = (freq-C)*np.pi/1000
    #omega = np.linspace(-1*np.pi, np.pi, 1000)
    ss = sigma**2
    exponential = np.exp(-1*complex(0, 1)*omega)
    denominator = np.abs(1 - alpha*exponential)**2
    return ss/denominator


def velocity_to_mode(k, l, vx, vy, d=8):
    return (k*vx + l*vy)/d

def find_velocity(k1, k2, l1, l2, freq_1, freq_2, d=8):
    
    modes = np.array([[k1, l1], [k2, l2]])
    print(modes)
    freq = d*np.array([freq_1, freq_2]).reshape(2, 1)
    inv = np.linalg.inv(modes)
    v = np.matmul(inv, freq)

    return v


freq_10_11 = [-38.617, 5.344, -9.672, -3.332, 32.404]
freq_19_19 = [-71.179, 9.902, -19.981, -5.120, 57.004]
freq_12_12 = [44.955, -6.253, 12.619, 3.233, -36.002]
freq_12_12_neg = -1*np.array(freq_12_12) 
freq_19_19neg = [-27.323, -4.829, 52.0521, -19.109, 34.252]

vx, vy = find_velocity(12, 10, 12, 11, -44.955, -38.617)

vx_known = [-20.737, 3.101, -15.165, 2.945, 4.790]
vy_known = [-9.233, 1.068, 6.752, -5.101, 19.212]


# read in data and pull out modes

infile = 'modes=32_frames=120000_1_franken_2khz_infinite_48x_training.fits'
#infile = 'data/modes=48_frames=60000_poyneer_8_franken_2khz_include_mode0.fits' 
hdu = fits.open(infile)
modes = np.complex(0, 1)*hdu[0].data + hdu[1].data

#mode_index, mode = 1618, (10,11) #(10, 11)

#mode_index = 539 # (-12, -12)
mode_index, mode = 1715, (12, 12)
#mode_index, mode = 2058, (19,19) # (19,19)
#mode_index = 196 # (19,-19)
#mode_index, mode = 1271, (3, 0)
#mode_index, mode = 1535, (8, 24)
freqs = []
#for index, vx_val in enumerate(vx_known):
#    freq = velocity_to_mode(mode[0], mode[1], vx_val, vy_known[index])
#    freqs.append(freq)

fs = 2e3
window = signal.get_window(window='hamming', Nx=4096)
#f, Pxx_den = signal.welch(modes[mode_index, :], fs=fs, window=window)
#sf, sP = zip(*sorted(zip(f, Pxx_den)))
#sf, sP = np.array(sf), np.array(sP)


#peaks = signal.find_peaks(sP, height=np.std(sP)*5)[0]

#plt.semilogy(sf, sP, color='cyan')
#for peak in peaks:
#    plt.axvline(sf[peak], linestyle='--', color='gray')
#for freq in freq_12_12:
#    plt.axvline(freq, linestyle='--', color='black')

#peak_freqs = np.array([sf[peak] for peak in peaks])
#physical_peaks = []
#for freq in freqs:
#    diff = np.abs(peak_freqs - freq)
#    if np.min(diff) < 3:
#        peak_index = np.where(np.min(diff) == diff)[0][0]
#        print(freq, peak_freqs[peak_index], np.min(diff), peaks[peak_index])
#        physical_peaks.append(peaks[peak_index])

"""
alphas, sigmas = [], []
#print(physical_peaks)
for peak in peaks:
    C = sf[peak]
    popt, pcov = curve_fit(perfect_peak, sf[peak-8:peak+8], sP[peak-8:peak+8], p0=[1, 0.995])
    sigma, alpha = popt
    if alpha > 0.99:
        sigmas.append(sigma)
        alphas.append(alpha)
        print(f'At {C}: sigma = {sigma}, alpha = {alpha}.')
        #plt.semilogy(sf[peak-25:peak+25], sP[peak-25:peak+25], color='black')
        plt.semilogy(sf[peak-100:peak+100], perfect_peak(sf[peak-100:peak+100], sigma, alpha), color='green')
    else:
        print(f'At {C}, bad fit: sigma = {sigma}, alpha = {alpha}.')
        plt.semilogy(sf[peak-100:peak+100], perfect_peak(sf[peak-100:peak+100], sigma, alpha), color='blue')
plt.show()
print('Alphas: ', alphas)
print('Sigmas: ', sigmas)
"""


infile = '../modes=32_frames=120000_1_franken_2khz_infinite_48x_training.fits'
window = signal.get_window(window='hamming', Nx=4000)
mode_data = ascii.read('mode_dict_32x.csv')
n_frames = 10000
fs = 2e3
mode_psds = {}
mode_layers = {}
for n_frames in [120000]:
    print(f'Running over {n_frames}:')
    for i, mode_index in enumerate(mode_data['mode_n']):
        print(mode_index)

        mode = (mode_data['k'][i], mode_data['l'][i])
        f, Pxx_den = signal.welch(modes[mode_index, :n_frames], fs=fs, window=window)
        sf, sP = zip(*sorted(zip(f, Pxx_den)))
        sf, sP = np.array(sf), np.array(sP)

        peaks = signal.find_peaks(sP, height=np.std(sP)*5)[0]

        #plt.semilogy(sf, sP, color='cyan')
        #for peak in peaks:
            #plt.axvline(sf[peak], linestyle='--', color='gray')

        alphas_real, alphas_imag, sigmas = [], [], []
        #print(physical_peaks)
        for peak in peaks:
            C = sf[peak]
            if np.abs(C) < 100:
                try:
                    popt, pcov = curve_fit(perfect_peak, sf[peak-8:peak+8], sP[peak-8:peak+8], p0=[1, 0.995])
                    sigma, alpha = popt
                    if alpha > 0.99:
                        sigmas.append(sigma)
                        alphas_real.append(alpha)
                        alphas_imag.append(-1*C)
                        #print(f'At {C}: sigma = {sigma}, alpha = {alpha}.')
                        #plt.semilogy(sf[peak-25:peak+25], sP[peak-25:peak+25], color='black')
                        #plt.semilogy(sf[peak-100:peak+100], perfect_peak(sf[peak-100:peak+100], sigma, alpha), color='green')
                    else:
                        print(f'At {C}, bad fit: sigma = {sigma}, alpha = {alpha}.')
                        #plt.semilogy(sf[peak-100:peak+100], perfect_peak(sf[peak-100:peak+100], sigma, alpha), color='blue')
                except RuntimeError:
                    print(f'At {C}, curve_fit could not converge.')
        print('Alphas Real: ', alphas_real)
        print('Alphas Imag: ', alphas_imag)
        print('Sigmas: ', sigmas)
        alpha_length_real = np.zeros(100)
        alpha_length_real[:len(alphas_real)] = alphas_real
        alpha_length_imag = np.zeros(100)
        alpha_length_imag[:len(alphas_imag)] = alphas_imag
        sigma_length = np.zeros(100)
        sigma_length[:len(sigmas)] = sigmas
        #plt.show()
        mode_layers[f'{str(mode_index)}_alphas_real'] = alpha_length_real
        mode_layers[f'{str(mode_index)}_alphas_imag'] = alpha_length_imag
        mode_layers[f'{str(mode_index)}_sigmas'] = sigma_length
        mode_psds[f'{str(mode_index)}_freq'] = sf
        mode_psds[f'{str(mode_index)}_power'] = sP

    #print(mode_layers)
    #print(mode_psds)
    ascii.write(mode_layers, f'mode_layers_training={n_frames}.csv', overwrite=True)
    #ascii.write(mode_psds, 'mode_psds_10000_complex.csv', overwrite=True)
