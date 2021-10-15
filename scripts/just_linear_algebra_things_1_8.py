""" Working title bc I'm the worst.

Sorry not sorry. 

Main structure: 
    - take n farmes and save open loop wfs data
    - make history vector h
    - l I n E a R  a L g E b r A

### STARTING FROM :
    m x i data
    sampled every i steps 
    with dt between each i 


"""
## -- IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from hcipy import *

## -- FUCTIONS

def calculate_DM_command(F, h, CM=1, gain=1):
    
    #dm_commands = gain * CM.dot(np.matmul(F, h))
    #print('F/h shapes: ', F.shape, h.shape)
    dm_commands = np.matmul(F, h)

    return dm_commands


def build_F(D, P):

    D_t = D.transpose()
    D_t_inverse = inverse_tikhonov(D_t, rcond=1e-3)
    #D_t_inverse = np.linalg.pinv(D_t, rcond=1e-3)
    P_t = P.transpose()
    F = np.matmul(D_t_inverse, P_t).transpose()

    return F

def build_atmosphere(wavelength, pupil_grid, model='single'):

    def build_multilayer_model(input_grid):
        #based off of https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf
        #heights = np.array([0.0, 2.1, 4.1, 6.5, 9.0, 12.0, 14.8])*1000
        #velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
        #outer_scales = np.array([20,20,20,20,20,20,20])
        #cn_squared = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027])* 1e-12
        
        heights = np.array([500, 1000, 2000, 4000, 8000, 16000])
        velocities = np.array([6.5, 6.55, 6.6, 6.7, 22, 9.5, 5.6])
        outer_scales = np.array([2, 20, 20, 20, 30, 40, 40])
        integrated_cn_squared = Cn_squared_from_fried_parameter(0.20, wavelength=wavelength)
        cn_squared = np.array([0.672, 0.051, 0.028, 0.106, 0.08, 0.052,0.012])*integrated_cn_squared
        
        layers = []
        for h, v, cn, L0 in zip(heights, velocities, cn_squared,outer_scales):
            layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))
        return layers 

    if model == 'single':
        cn_squared = Cn_squared_from_fried_parameter(0.20, wavelength=wavelength)
        layer = InfiniteAtmosphericLayer(pupil_grid, cn_squared, 20, 7)
    
    elif model == 'multilayer':
        layer = MultiLayerAtmosphere(build_multilayer_model(pupil_grid))

    return layer


def convert_phase_to_wfe(phase, wavelength=658e-9, unit_conversion=1e9):
    
    wfe = phase*(wavelength*unit_conversion)/(2*np.pi)
    
    return wfe 

def build_sample_data(i, j, k, n_iters, data_type='sin', save=None):

    past = np.ones((i, j, k))
    future = np.ones((i, j, n_iters))
    for index in range(n_iters):
        if data_type == 'sin':
            future[:, :, index] = np.sin(2*np.pi*index/20)
            if index < k:
                past[:, :, index] = np.sin(2*np.pi*index/20)

        if data_type == 'linear':
            future[:, :, index] += 1.005
            if index < k:
                past[:, :, index] += 1.005
        
    if data_type == 'AO':
        pupil_grid = make_pupil_grid(i, 8)
        #wavelength = 1.63e-06
        wavelength = 658e-9
        layer = build_atmosphere(wavelength, pupil_grid, model='single')
        
        # running at kHz timescale
        for iters in [(k, past), (n_iters, future)]:
            for index in range(iters[0]):
                layer.t = 0.001*(index+1)
                phase = layer.phase_for(wavelength).shaped
                iters[1][:, :, index] = phase
    
    if data_type == '2D sin':
        for time_index in range(n_iters):
            for column_index in range(j):
                future[:, column_index, time_index] *= np.sin(2*np.pi/20*(column_index+time_index*.5))
                if time_index < k:
                    past[:, column_index, time_index] *= np.sin(2*np.pi/20*(column_index+time_index*.5))
    
    if data_type == 'T/T':
        for iters in [(k, past), (n_iters, future)]:
            for time_index in range(iters[0]):
                # this doesn't need to be a loop...
                iters[1][:, :, time_index] = np.array([random.random()*12-6, random.random*12-6])
    
    if save is not None:
        hdu_list = fits.HDUList([fits.PrimaryHDU(past), fits.ImageHDU(future)])
        hdu_list.writeto(save)
    return past, future


def pseudo_inverse_least_squares(D, P, alpha=1):

    D_t = D.transpose()
    identity_matrix = np.identity(np.shape(D)[0])
    
    F = np.matmul(np.matmul(P, D_t), np.linalg.inv(np.matmul(D, D_t) + alpha*identity_matrix))

    return F


## -- FIDDLING
t_start = time.time()
# given some 3d numpy array that holds WFS data for some ixj actuators over some
# k timesteps
i, j = 40, 40
k = 60000
n_iters = 60000

past, future =  build_sample_data(i, j, k, n_iters, data_type='AO')
t_data = time.time()
print(f'Data simulated after {t_data - t_start} seconds.') 
# first flatten her out so it's only over M actuators/subapertures
m = i*j
#flat_wfs = past_dummy_data.reshape((m, k))
flat_wfs = past.reshape((m, k))

# now decide how long the history vector is and how long training matrix is
n = 12 # MUST BE GREATER THAN DT YOU DINGDONG
dt = 2
l = k-n-dt # max k-n
# now create m*x length history vector and m*n x l length training matrix
training_matrix = np.zeros((m*n, l))
for training_index, max_index in enumerate(np.flip(np.arange(n+1, l+n+1))):
    min_index = max_index - n
    #print(min_index, max_index)
    data_slice = np.fliplr(flat_wfs[:, min_index:max_index]).transpose()
    #print(training_matrix[:, max_index].shape, data_slice.flatten().shape)
    training_matrix[:,training_index] = data_slice.flatten()

# now create state 1 x l state matrix at time delay 3
state_matrix = np.zeros((m,l))
#for state_index in range(l):
for state_index, index in enumerate(np.flip(np.arange(n, l+n))):
    wfs_index = index+dt
    state_matrix[:, state_index] = flat_wfs[:, wfs_index]

# build F matrix with predictive filter
#F = build_F(training_matrix, state_matrix)
F = pseudo_inverse_least_squares(training_matrix, state_matrix)
t_filter = time.time()
print(f'Filter created after {t_filter - t_data} seconds.')

#flat_wfs_future = future_dummy_data.reshape((i*j, n_iters))
flat_wfs_future = future.reshape((i*j, n_iters))
flat_wfs_prediction = np.zeros_like(flat_wfs_future)
rms_future = []
prediction = []
quasi_integrator = []
rms_pred = []
rms_int = []
for max_index in range(n+1, n_iters-dt):
    min_index = max_index-n
    data_slice = np.fliplr(flat_wfs_future[:, min_index+1:max_index+1]).transpose()
    h = data_slice.flatten()
    dm_commands = calculate_DM_command(F,h)
    #print(flat_wfs_prediction.shape, dm_commands.shape) 
    flat_wfs_prediction[:, max_index+dt] = dm_commands
    #print(f'At iter {max_index}: {np.median(dm_commands.reshape((10,10)))} vs {np.median(flat_wfs_future[:, max_index+dt])}')
    rms_future.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt])**2)))
    rms_pred.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt] - dm_commands)**2)))
    rms_int.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt] - flat_wfs_future[:, max_index])**2)))

#plt.plot(future, label='future')
#plt.plot(prediction, label='prediction', linestyle='--')
#plt.plot(quasi_integrator, label='quasi integrator', linestyle='-.')
#plt.legend()
#plt.yscale('log')
#plt.savefig('predictive_test_AO.png')
#plt.show()
#plt.clf()
t_end = time.time()
print(f'Predictor loop completed after {t_end - t_filter} seconds.')

plt.plot(np.array(rms_future)*1e-3, label='uncorrected', color='gray')
#plt.plot(np.array(rms_future), label='uncorrected', color='gray')
plt.plot(np.array(rms_int)*1e-3, label='quasi integrator', color='green')
#plt.plot(np.array(rms_int), label='quasi integrator', color='green')
plt.plot(np.array(rms_pred)*1e-3, label='prediction', color='cyan')
#plt.plot(np.array(rms_pred), label='prediction', color='cyan')
plt.legend()
plt.xlabel('Iterations (1kHz)')
plt.ylabel('RMS wavefront error [um]') 
plt.yscale('log')
#plt.ylim(1e-3, 10)
plt.savefig('predictive_test_1_layer_8m.png')
#plt.show()
plt.clf()

#print(f'Uncorrected: {np.median(rms_future)}, quasi-integrator: {np.median(rms_int)}, predictor: {np.median(rms_pred)} nm.')
print(f'Uncorrected: {np.median(rms_future)}, quasi-integrator: {np.median(rms_int)}, predictor: {np.median(rms_pred)} nm.')

#plt.hist(rms_int, bins=100, label='quasi integrator', alpha=.3, color='gray')
#plt.hist(rms_pred, bins=100, label='prediction', alpha=.3, color='green')
#plt.legend()
#plt.savefig('predictive_test_histogram_TT.png')
#plt.show()
