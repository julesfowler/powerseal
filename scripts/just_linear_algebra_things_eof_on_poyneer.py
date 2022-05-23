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
from astropy.io import fits
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
            layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h))
        return layers 

    if model == 'single':
        cn_squared = Cn_squared_from_fried_parameter(0.20, wavelength=wavelength)
        layer = InfiniteAtmosphericLayer(pupil_grid, cn_squared, 20, 7)
    
    elif model == 'multilayer':
        layer = MultiLayerAtmosphere(build_multilayer_model(pupil_grid))

    return layer


def convert_phase_to_wfe(phase, wavelength=500e-9, unit_conversion=1e9):
    
    wfe = phase*(wavelength*unit_conversion)/(2*np.pi)
    
    return wfe 


def apply_binning(data, resolution, modes):

    # taken almost exactly from Maaike's notebook

    if resolution==modes:
        binned_data = data

    else:
        binned_data = np.zeros((modes, modes))
        binning_factor = int(resolution/modes)
        for i_index in np.arange(binning_factor-1, resolution-binning_factor, binning_factor):
            for j_index in np.arange(binning_factor-1, resolution-binning_factor, binning_factor):
                binned_col = np.mean( 
                             data[i_index-int((binning_factor-1)/2):i_index+int((binning_factor-1)/2),
                                  j_index-int((binning_factor-1)/2):j_index+int((binning_factor-1)/2)]
                             )

                binned_data[int((i_index+1)/binning_factor-1), int((j_index+1)/binning_factor-1)] = binned_col

    return binned_data


def misc_indexing_thoughts(wfs_state, mask, wfs_corrected):

    # given some array of data and some mask
    full_index = np.arange(len(wfs_state.flatten()))

    wfs_flat = wfs_state[mask]
    index_flat = full_index[mask.flatten()]
    
    # pass around these two, and index flat will
    # let you reassign later with
    wfs_reshape  = np.zeros_like(wfs_state)
    np.put(wfs_reshape, index_flat, wfs_corrected)
    
    # now this holdes the reshaped corrected values whee
    return wfs_reshape


def pupil(N):
    p = np.zeros([N,N])
    radius = N/2.
    [X,Y] = np.meshgrid(np.linspace(-(N-1)/2.,(N-1)/2.,N),np.linspace(-(N-1)/2.,(N-1)/2.,N))
    R = np.sqrt(pow(X,2)+pow(Y,2))
    p[R<=radius] = 1
    return p



def read_data_square(k, n_iters, wf_path='/Users/julesfowler/Downloads/residualWF.npy',
              dm_path='/Users/julesfowler/Downloads/dmc.npy'):

    dm_commands = np.load(dm_path)
    wf_residuals = np.load(wf_path)

    integrator_phase = wf_residuals*0.6
    open_loop_phase = (-1*dm_commands + wf_residuals)*0.6
    open_loop_phase = open_loop_phase[:, 4:17, 4:17]

    past = open_loop_phase[:k, :, :].transpose()
    future = open_loop_phase[k:k+n_iters, :, :].transpose()
    future_integrator_residuals = integrator_phase[k:k+n_iters, :, :].transpose()
    future_integrator_residuals = future_integrator_residuals[4:17, 4:17, :] 

    
    test = future[:, :, 35]
    print(np.median(test))
    #print(np.median(future[35]))
    print(np.median(test[test != 0]))
    print(np.shape(past), np.shape(future), np.shape(future_integrator_residuals))
    return past, future, future_integrator_residuals


def read_data(k, n_iters, wf_path='/Users/julesfowler/Downloads/residualWF.npy',
              dm_path='/Users/julesfowler/Downloads/dmc.npy'):

    dm_commands = np.load(dm_path)
    dm_commands[dm_commands > 10] = 10
    dm_commands[dm_commands < -10] = -10
    pupil_mask = np.array([pupil(21)], dtype=bool)
    x, y = pupil_mask.shape[1], pupil_mask.shape[2] 
    
    wf_residuals = np.load(wf_path)
    integrator_phase = wf_residuals*0.6
    open_loop_phase = (-1*dm_commands + wf_residuals)*0.6
    past_shaped = open_loop_phase[:k, :, :].transpose()
    future_shaped = open_loop_phase[k:k+n_iters, :, :].transpose()
    future_residuals = integrator_phase[k:k+n_iters, :, :].transpose()
    
    print(x, y)
    print(pupil_mask.shape)

    past_mask = np.broadcast_to(pupil_mask.reshape(y, x, 1), (y, x, k))
    full_index_past = np.arange(k*x*y)
    index_flat_past = full_index_past[past_mask.flatten()]

    future_mask = np.broadcast_to(pupil_mask.reshape(y, x, 1), (y, x, n_iters))
    full_index_future = np.arange(n_iters*x*y)
    index_flat_future = full_index_future[future_mask.flatten()]
    

    past = past_shaped[past_mask]
    future = future_shaped[future_mask]
    future_integrator_residuals = future_residuals[future_mask]
    
    test = future_shaped[:, :, 35]
    print(np.median(test))
    print(np.median(future[35]))
    print(np.median(test[test != 0]))

    return past, index_flat_past, future, future_integrator_residuals, index_flat_future 


def build_sample_data(i, j, k, n_iters, data_type='sin', save=None, resolution=None):
    
    resolution = i if resolution is None else resolution

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
        pupil_grid = make_pupil_grid(resolution, 8)
        science_wavelength = 1.63e-06
        wavelength = 658e-9
        layer = build_atmosphere(wavelength, pupil_grid, model='single')
        
        # running at kHz timescale
        for iters in [(k, past), (n_iters, future)]:
            for index in range(iters[0]):
                layer.t = 0.001*(index+1)
                phase = layer.phase_for(science_wavelength).shaped
                iters[1][:, :, index] = apply_binning(phase, resolution, i)
    
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
        hdu_list.writeto(save, overwrite=True)
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
resolution = 144
i, j = 13, 13
n_iters = 30000

#hdu = fits.open('turbulence_1_8_franken_AOres=48_2khz_infinite_500nm_120000.fits')
#hdu = fits.open('turbulence_1_8_ptt_AOres=48_2khz_infinite_500nm_both.fits')
#future_data = hdu[1].data
#past_data = hdu[0].data
n_iters = 10000
k = 30000
predictor_avg_rms = []
for n in [4]:
    #n = 4 # MUST BE GREATER THAN DT YOU DINGDONG
    #past = np.zeros((i, j, k))
    #future = np.zeros((i, j, n_iters))
    #for index in range(k):
    #    print(np.shape(data[:, :, index]), resolution, i
        #past[:, :, index] = apply_binning(past_data[24:120, 24:120, index], 96, 32)
        #past[:, :, index] = past_data[8:40, 8:40, index]
        #if index < n_iters:
            #future[:, :, index] = apply_binning(future_data[24:120, 24:120, index], 96, 32)
            #future[:, :, index] = future_data[8:40, 8:40, index]
    #past, future = build_sample_data(i, j, k, n_iters, resolution=resolution, data_type='AO', save='turbulence.fits')
    #test_data = fits.open('turbulence.fits')
    #past, future = test_data[0].data, test_data[1].data
    past, future, integrator_residuals = read_data_square(k, n_iters)
    #print(np.mean(past), np.mean(future))
    #print(np.sqrt(np.mean(past**2)), np.sqrt(np.mean(future**2)))
    t_data = time.time()
    print(f'Data simulated after {t_data - t_start} seconds.') 
    # first flatten her out so it's only over M actuators/subapertures
    #m = 349
    m = i*j
    #flat_wfs = past_dummy_data.reshape((m, k))
    flat_wfs = past.reshape((m, k))
    #flat_wfs = past.reshape((k, m))

    # now decide how long the history vector is and how long training matrix is
    dt = 1
    l = k-n-dt # max k-n
    # now create m*x length history vector and m*n x l length training matrix
    training_matrix = np.zeros((m*n, l))
    for training_index, max_index in enumerate(np.flip(np.arange(n+1, l+n+1))):
        min_index = max_index - n
        #print(min_index, max_index)
        data_slice = np.fliplr(flat_wfs[:, min_index:max_index]).transpose()
        #print(data_slice.shape)
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
    flat_wfs_future = future.reshape((m, n_iters))[:, :n_iters]
    #flat_integrator_residuals = np.zeros_like(flat_wfs_future) 
    flat_integrator_residuals = integrator_residuals.reshape((m, n_iters))
    flat_wfs_prediction = np.zeros_like(flat_wfs_future)
    rms_future = []
    prediction = []
    quasi_integrator = []
    rms_pred = []
    rms_int = []
    for max_index in range(n+1, n_iters-dt):
        print(f'Iter {max_index} of {n_iters}')
        min_index = max_index-n
        data_slice = np.fliplr(flat_wfs_future[:, min_index+1:max_index+1]).transpose()
        h = data_slice.flatten()
        dm_commands = calculate_DM_command(F,h)
        flat_wfs_prediction[:, max_index+dt] = dm_commands
        #flat_integrator_residuals = flat_wfs_future[:, max_index+dt] - flat_wfs_future[:, max_index]
        #print(f'At iter {max_index}: {np.median(dm_commands.reshape((10,10)))} vs {np.median(flat_wfs_future[:, max_index+dt])}')
        
        #rms_future.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt])**2)))
        rms_future.append(np.sqrt(np.mean((flat_wfs_future[:, max_index+dt])**2)))
        #rms_pred.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt] - dm_commands)**2)))
        rms_pred.append(np.sqrt(np.mean((flat_wfs_future[:, max_index+dt] - dm_commands)**2)))
        #rms_int.append(np.sqrt(np.mean(convert_phase_to_wfe(flat_wfs_future[:, max_index+dt] - flat_wfs_future[:, max_index])**2)))
        rms_int.append(np.sqrt(np.mean((flat_integrator_residuals[:, max_index+dt])**2)))


    save = f'eof_prediction_dt={dt}_k={k}_n={n}_telemetry.fits'
    hdu_list = fits.HDUList([fits.PrimaryHDU(flat_wfs_future),
        fits.ImageHDU(flat_wfs_prediction), fits.ImageHDU(flat_integrator_residuals)])
    hdu_list.writeto(save, overwrite=True)

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

    plt.plot(np.array(rms_future)*1e3, label='uncorrected', color='gray')
    #plt.plot(np.array(rms_future), label='uncorrected', color='gray')
    plt.plot(np.array(rms_int)*1e3, label='integrator', color='green')
    #plt.plot(np.array(rms_int), label='quasi integrator', color='green')
    plt.plot(np.array(rms_pred)*1e3, label='prediction', color='cyan')
    #plt.plot(np.array(rms_pred), label='prediction', color='cyan')
    plt.legend()
    plt.xlabel('Iterations (1kHz)')
    plt.ylabel('RMS wavefront error [nm]') 
    plt.yscale('log')
    plt.title(f'EOF RMS = {round(np.median(rms_pred)*1e3, 3)} +/- {round(np.std(rms_pred)*1e3, 3)}nm')
    #plt.ylim(1e-2, 10)
    plt.savefig(f'eof_dt={dt}_k={k}_n={n}_square_telemetry.png')
    #plt.show()
    plt.clf()

    #print(f'Uncorrected: {np.median(rms_future)}, quasi-integrator: {np.median(rms_int)}, predictor: {np.median(rms_pred)} nm.')
    print(f"For k={k}, n={n}:")
    print(f'Uncorrected: {np.median(rms_future)*1e3}, integrator: {np.median(rms_int)*1e3}, predictor: {np.median(rms_pred)*1e3} nm.')
    predictor_avg_rms.append(np.median(rms_pred))

#plt.plot([45000, 50000, 55000, 60000, 65000], predictor_avg_rms)
#plt.xlabel('training data length [n frames]')
#plt.ylabel('RMS [nm]')
#plt.show()
#plt.hist(rms_int, bins=100, label='quasi integrator', alpha=.3, color='gray')
#plt.hist(rms_pred, bins=100, label='prediction', alpha=.3, color='green')
#plt.legend()
#plt.savefig('predictive_test_histogram_TT.png')
#plt.show()
