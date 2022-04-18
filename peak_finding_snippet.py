## - snippet / spaghetti code for peak ID


from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# read in data and pull out modes
infile = ...
hdu = fits.open(infile)
modes = np.complex(0, 1)*hdu[0].data + hdu[1].data


window = signal.get_window(window='hamming', Nx=1000)
f, Pxx_den = signal.welch(modes[1715, :], fs=fs, window=window) # the 12 x 12 mode
sf, sP = zip(*sorted(zip(f, Pxx_den)))
sf, sP = np.array(sf), np.array(sP)



def a(t, modes, index, noise, alpha):
    return alpha*modes[index, t-1] + noise

def peak_a(autocorrelation_func, alpha, omega):
    return (np.std(autocorrelation_func)/np.abs(1 - alpha*np.exp(np.complex(0, 1)*omega)))**2

peaks = signal.find_peaks(sP, height=np.std(sP)*5)[0]
peak_width = 5

for peak in peaks:
    # isolate peak region
    # calculate peak power 
    # subtract out peak -- might not actualy need to do this?

