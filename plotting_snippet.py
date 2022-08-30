from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# NOTE I'm using a convenience animation helper from 
# my favorite optics/High Contrast Imaging Python package
# you'll probably need to install it with pip3 install hcipy
from hcipy import GifWriter

def convert_phase_to_wfe(phase, wavelength=1.63e-6, unit_conversion=1e9):
    """ Given some phase error in radians, convert to wavefront error."""

    wfe = phase*(wavelength*unit_conversion)/(2*np.pi)
    return wfe

# Read in the data 
# Simulation data was created with HCIPy
# Predictive control data I generated for my research
# Happy to chat about this but right now that code is 
# pure spaghetti and non-trivial to clean up and share :) 

# Read in the data from my results file
predictive_data = fits.open('data/prediction_results.fits')

# Split these into the actual state of the atmosphere and 
# What our predictive controller thinks it will be
future, prediction = predictive_data[0].data, predictive_data[1].data

# Due to math being mathy, this data comes in flat, so reshape it to be shaped
# like the images 
future_shaped = future.reshape((20,20,30000))
predictor_shaped = prediction.reshape((20,20,30000))


# Set up animation writer and some arrays to update
anim = GifWriter('AO_final.gif', framerate=6)
x1 = []
y1, y2, y3 = [], [], []

# These map to properties of the simulation 
# N is number of associated frames (something something linear algebra
# predictor)
# dt is the timestep lag -- set to 1ms for this simulation
n=5
dt=1

# Loop through the 30000 exposures (1/2 minute of data in real time!)
# Note that at least on my machine trying to do any more than every 100th
# exposure totally killed my computer. Even like this is took a good 15 minutes
# to write out.
for max_index in range(6, 29907, 100):
    min_index = max_index-n
    print(max_index, ' frame added')

    # Set up figure
    fig, ax = plt.subplots(2, 3, gridspec_kw={'height_ratios': [3,1]}, figsize=(17, 10))
    fig.tight_layout()
    
    # First subplot holds atmospheric errors
    ax[0,0].imshow(future_shaped[:19,:19,max_index+dt], vmin=-7, vmax=7)
    ax[0,0].axis('off')
    
    # Second subplot we subtract the atmosphere with a 1 frame delay
    # I.e., how much error we'd get if we had a perfect correction on a 1 dt
    # time delay
    ax[0,1].imshow(future_shaped[:19,:19,max_index+dt]-future_shaped[:19,:19,max_index], vmin=-.07, vmax=.07)
    ax[0,1].axis('off')
    
    # And how closely we can predict the atmospheric errors
    ax[0,2].imshow(future_shaped[:19,:19,max_index+dt]-predictor_shaped[:19,:19,max_index+dt], vmin=-.07, vmax=.07)
    ax[0,2].axis('off')
    
    # Do some trickery to make this a running arry we update one point at a time
    x1.append(max_index)
    # This is calculating the root mean square (RMS) error associated with these
    # images for a numerical comparison
    y1.append(np.sqrt(np.mean(convert_phase_to_wfe(future[:,max_index+dt])**2))*1e3)
    y2.append(np.sqrt(np.mean(convert_phase_to_wfe(future[:,max_index+dt]-future[:,max_index])**2))*1e3)
    y3.append(np.sqrt(np.mean(convert_phase_to_wfe(future[:,max_index+dt]-prediction[:,max_index+dt])**2))*1e3)
    
    # And then we plot out each of these RMS error plots
    ax[1,0].set_ylabel('RMS Wavefront Error [um]')
    ax[1,0].set_title('Uncorrected Turbulence (200x scale)')
    ax[1,0].plot(x1, y1, color='gray')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlim(-100, 29950)
    ax[1,0].set_ylim(400, 1e7)
    
    ax[1,1].set_title('Traditional Integrator')
    ax[1,1].plot(x1, y2, color='green')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlim(-100, 29950)
    ax[1,1].set_ylim(400, 1e7)
    
    ax[1,2].set_title('Predictive Control')
    ax[1,2].set_yscale('log')
    ax[1,2].plot(x1, y3, color='cyan')
    ax[1,2].set_xlim(-100, 29950)
    ax[1,2].set_ylim(400, 1e7)
    
    # Finally we add the single frame
    anim.add_frame()

# And closing the writer saves it.
anim.close()
