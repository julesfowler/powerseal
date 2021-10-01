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
    P_t = P.transpose()
    F = np.matmul(D_t_inverse, P_t).transpose()

    return F


def build_data(i, j, k, n_iters, data_type='sin'):

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
                pas[:, :, index] += 1.005
        
        if data_type == 'AO':
            raise NotImplementedError

    return past, future

## -- FIDDLING

# given some 3d numpy array that holds WFS data for some ixj actuators over some
# k timesteps
i, j = 10, 10
k = 500
past_dummy_data = np.ones((i,j,k))
for index in range(1, k):
    past_dummy_data[:, :, index] = past_dummy_data[:, :, index]*np.sin(2*np.pi*index/20)

# first flatten her out so it's only over M actuators/subapertures
m = i*j
flat_wfs = past_dummy_data.reshape((m, k))

# now decide how long the history vector is and how long training matrix is
n = 10
l = 450 # max k-n
# now create m*x length history vector and m*n x l length training matrix
training_matrix = np.zeros((m*n, l))
min_index = 0
for max_index in range(n, l):
    #print(min_index, max_index)
    data_slice = flat_wfs[:, min_index:max_index]
    h = data_slice.flatten()
    #print(training_matrix[:, max_index].shape, data_slice.flatten().shape)
    training_matrix[:, max_index] = data_slice.flatten()
    min_index += 1

# now create state 1 x l state matrix at time delay 3
dt = 3
state_matrix = np.zeros((m,l))
for state_index in range(l):
    wfs_index = state_index+dt
    state_matrix[:, state_index] = flat_wfs[:, wfs_index]

# build F matrix with predictive filter
F = build_F(training_matrix, state_matrix)

# now run a stripped down AO loop on some future dummy data
n_iters = int(10000)
future_dummy_data = np.ones((i, j, n_iters))
for index in range(1, n_iters):
    future_dummy_data[:, :, index] = future_dummy_data[:, :, index]*np.sin(2*np.pi*index/20)

flat_wfs_future = future_dummy_data.reshape((i*j, n_iters))
flat_wfs_prediction = np.zeros_like(flat_wfs_future)
min_index = 0
future = []
prediction = []
quasi_integrator = []
for max_index in range(n, n_iters-dt):
    data_slice = flat_wfs_future[:, min_index:max_index]
    h = data_slice.flatten()
    dm_commands = calculate_DM_command(F,h)
    min_index += 1
    #print(flat_wfs_prediction.shape, dm_commands.shape) 
    flat_wfs_prediction[:, max_index+dt] = dm_commands
    print(f'At iter {max_index}: {np.median(dm_commands.reshape((10,10)))} vs {np.median(flat_wfs_future[:, max_index+dt])}')
    future.append(np.median(flat_wfs_future[:, max_index+dt]))
    prediction.append(np.median(dm_commands))
    quasi_integrator.append(np.median(flat_wfs_future[:, max_index]))

plt.plot(future, label='future')
plt.plot(prediction, label='prediction', linestyle='--')
plt.plot(quasi_integrator, label='quasi integrator', linestyle='-.')
plt.legend()
#plt.yscale('log')
plt.savefig('predictive_test_sin.png')
plt.show()
plt.clf()
plt.plot(np.array(future)-np.array(prediction), label='prediction')
plt.plot(np.array(future)-np.array(quasi_integrator), label='quasi integrator')
plt.legend()
plt.savefig('predictive_test_sin_residuals.png')
plt.show()
