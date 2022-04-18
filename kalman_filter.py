## -- IMPORTS
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy


## -- FUNCTIONS


def build_initial_filter(A, B, C, P_w, P_v):
    """ Builds the 0th state for the Kalman Filter Predictor. 

    Parameters
    ----------
    A : np.array 
        (L+6) x (L+6) (initial) filter matrix
    B : np.array 
        (L+6) x (L+1) (initial) state space model matrix
    C : np.array
        1 x (L+6) (initial) control law matrix
    P_w : np.array
        (L+1) x (L+1) (initial) covariance matrix 
    P_v : float
        1 x 1 scalar of the white noise 

    Returns
    -------
    F_0 : np.array
        (L+6) x (L+6) 0th order filter matrix
    P_0 : np.array
        (L+6) x (L+6) 0th order covariance matrix
    O_0 : np.array
        (L+6) x (L+6) 0th order control matrix 
    """

    B_herm = B.conj().T
    C_herm = C.conj().T

    F_0 = A
    P_0 = np.matmul(np.matmul(B, P_w), B_herm)
    O_0 = np.matmul(C_herm*(1/P_v), C)

    return F_0, P_0, O_0


def evolve_state_matrices(F, P, O, rcond=1e-6):
    """ Evolves the state forward to predict the next timestep. 

    Parameters
    ----------
    F : np.array
        (L+6) x (L+6) filter matrix at step t
    P : np.array
        (L+6) x (L+6) covariance matrix at step t
    O : np.array
        (L+6) x (L+6) control matrix at step t

    Returns
    -------
    F_next : np.array
        (L+6) x (L+6) filter matrix at step t+1
    P_next : np.array
        (L+6) x (L+6) covariance matrix at step t+1
    O_next : np.array 
        (L+6) x (L+6) control matrix at step t+1
    """

    T_term = np.matmul(P, O)
    size = term.shape[0] 
    T = np.pinv(np.identity(size, dtype=float) + term, rcond=rcond) 
    
    T_herm = T.conj().T
    F_herm = F.conj().T

    F_next = np.matmul(np.matmul(F, T), F)
    P_next = P + np.matmul(np.matmul(np.matmul(F, P), T_herm), F_herm)
    O_next = O + np.matmul(np.matmul(np.matmul(F_herm, O), T), F)

    return F_next, P_next, O_next

def evolve_state(x_prev, dm_prev, y, A, C, G, K): 

    l_plus_six = np.shape(K)[0]
    kalman_term = np.identity(l_plus_six) - np.matmul(K, C)
    #print('I - KC ', np.shape(kalman_term)) 
    x = np.matmul(np.matmul(kalman_term, A), x_prev) +  np.matmul(kalman_term, G)*dm_prev + K*y

    wf_next = x[l_plus_six-5]
    #print('x ', np.shape(x))
    #print(wf_next)
    #print(x)

    return x, wf_next, 


def build_A(alpha):
    """ A true monstrosity, that makes up the initial filter matrix. 
    
    R : diagonal matrix with alphas on the diagonal

    A = [ R {(L+1) x (L+1), 0 {(L+1) x 1}, 0.., 0.., 0..,  0 
          1 {1 x (L+1)},    0,             0,   0,   0,    0
          0 {1 x (L+1)],    1,             0,   0,   0,    0
          0..,              0,             1,   0,   0,    0
          0..,              0,             0,   0,   0,    0
          0..,              0,             0,   0,   1,    0] 

    Parameters
    ----------
    alpha : np.array
        (L+1) length array filled with alphas fit to wind layers

    Returns
    -------
    A : np.array
        (L+6) x (L+6) filter matrix. 
    """

    l_plus = len(alpha)
    l = l_plus - 1

    R = np.zeros((l_plus, l_plus))
    for index, alpha_val in enumerate(alpha):
        R[index, index] = alpha_val

    A = np.zeros((l+6, l+6))
    
    A[:l_plus, :l_plus] = R
    A[l_plus, :l_plus] = np.ones(l_plus)

    A[-1, -2] = 1
    A[-3, -4] = 1
    A[-4, -5] = 1

    return A

def build_autoregression(alpha_layers, w_layers, a_minus):
    
    a_vector = np.zeros(len(alpha_layers), dtype=complex)
    for index, alpha_value in enumerate(alpha):
        a_value = alpha_value*a_minus + w_layers[index]
        a_vector[index] = a_value

    return np.array(a_vector)


def build_B(l):
    """ B is (L+6) x (L+1) (initial) state space model matrix. 
    B = [I {(L+1) x (L+1)} 
         0 {5 x (L+1)}    ]

    Parameters
    ----------
    l : int
        There are L+1 layers in a given mode because of zero-indexing.
    
    Returns
    -------
    C : np.array
        1 x (L+6) (initial) control matrix 
    """
    
    B = np.zeros((l+6, l+1))
    B[:l+1, :l+1] = np.identity(l+1, dtype=float)

    return B


def build_C(l):
    """ C is 1 x (L+6) control matrix. 
    C = [0 {1 x (L_1)}, 0, 0, 1, -1, 0] 

    Parameters  
    ----------
    l : int
        There are L+1 layers in a given mode because of zero-indexing.
    """

    C = np.zeros(l+6)
    C[-2] = -1.0
    C[-3] = 1

    return C.reshape((1, l+6))


def build_G(l):
    """ G is 1 x (L+6) DM update matrix. 

    Parameters  
    ----------
    l : int
        There are L+1 layers in a given mode because of zero-indexing.
    
    G = [0 {1 x (L+1), 0, 0, 0, 1, 0]^T 

    Returns
    -------
    G : np.array    
        1 x (L+1) DM update matrix. 
    """
    
    G = np.zeros(l+6)
    G[-2] = 1

    return G.reshape((l+6, 1))


def build_K(P_s, C, P_v):
    """ Calculate steady state Kalman gains, a 1 x L+6 matrix."""
    
    term = 1/(np.matmul(np.matmul(C, P_s), C.transpose()))
    K = np.matmul(P_s, C.transpose())*term

    return K.reshape((l+6, 1))


def build_P_s(A, B, C, P_v, P_w):
    """ Solve the discrete algebraic ricatti equation for the steady state
    covariance matrix. """

    a = A.transpose()
    b = C.transpose()
    q = np.matmul(np.matmul(B, P_w), B.transpose())
    r = P_v
    
    P_s = scipy.linalg.solve_discrete_are(a, b, q, r)
    
    return P_s


def build_P_v(v):
    """ Variance of the white noise distribution. 
    
    Parameters
    ----------
    v : np.array
        White noise at some mode.

    Returns
    -------
    P_v : float
        Scalar variance of the white noise. 
    """

    P_v = np.var(v)
    
    return P_v


def build_P_w(sigma_squared):
    """ P_w is (L_1) x (L+1) diagonal matrix with sigma_squared for each layer
    on the diagonal. This is the covariance matrix -- or at least most of it. 
    P_w = [sigma_0**2 ...   0
           0 sigma_1**2 ... 0
           ...
           ...        sigma_L**2]"""
    
    l_plus = len(sigma_squared)
    P_w = np.zeros((l_plus, l_plus))

    for index, sigma_val in enumerate(sigma_squared):
        P_w[index, index] = sigma_val

    return P_w


def build_x_init(a, wavefront, dm, timestep):
    
    l_plus = len(a)
    l = l_plus - 1

    x = np.zeros(l+6, dtype=complex)
    x[:l_plus] = a
    x[-5] = wavefront[timestep+1]
    x[-4] = wavefront[timestep]
    x[-3] = wavefront[timestep-1]
    x[-2] = dm[timestep-1]
    x[-1] = dm[timestep-2]

    return x.reshape((l+6, 1))


def calculate_dm_command(wavefront, interaction_matrix=1):

    dm_command = wavefront.real

    return dm_command

## -- GET IT

infile = 'data/modes=48_frames=60000_poyneer_8_franken_2khz_include_mode0.fits'
hdu = fits.open(infile)
modes = np.complex(0, 1)*hdu[0].data + np.complex(1, 0)*hdu[1].data
wavefront = modes[1715, :]


# starting with 4 wind layers at mode 12 x 12
l = 4
sigma_squared = [7.1216722414696945e-06,
                 6.541757149605491e-06,
                 4.870870115743126e-06,
                 4.238942978009774e-06,
                 6.1543419962556235e-06]
alpha = [0.995823704690824,
         0.9960367335962748,
         0.9986817173944081,
         0.9986202334957323,
         0.9969061483950212]
v = ...

# perfect DM for now, so d[t] = x[t]
dt_start = 4
sample_length = 29000
timesteps = np.arange(sample_length)
wavefront = wavefront[:sample_length]
dm_commands = np.zeros(sample_length)
dm_commands[:3] = [wf.real for wf in wavefront[:3]]
sensor_noise_factor = np.median(wavefront.real)*1e-2
v = sensor_noise_factor*np.random.random(sample_length)*0
measurement_data = wavefront + v

A = build_A(alpha)
B = build_B(l)
C = build_C(l)
G = build_G(l)
P_w = build_P_w(sigma_squared)
P_v = build_P_v(v)
P_s = build_P_s(A, B, C, P_v, P_w)
K = build_K(P_s, C, P_v)

# calculate initial state vector 
#w_layers = np.zeros(l+1) # white noise set to zero for now
w_layers = np.random.random(l+1)*np.std(wavefront)
a_init = w_layers #build_autoregression(alpha, w_layers, wavefront[0])
x = build_x_init(a_init, wavefront, dm_commands, dt_start)

wf_predictions = np.zeros_like(wavefront)

for dt in timesteps:
    
    y = measurement_data[dt]
    dm_prev = dm_commands[dt-1]
    x, wf_prediction = evolve_state(x, dm_prev, y, A, C, G, K)
    #if dt < 28000:
        #print(wf_prediction, wavefront[dt+1])
    dm_command = calculate_dm_command(wf_prediction)

    wf_predictions[dt] = wf_prediction
    dm_commands[dt] = dm_command

#plt.plot(timesteps, measurement_data.real)
plt.title('Imaginary Coefficient')
plt.plot(timesteps, wavefront.imag, color='cyan', label='simulated data 12x12 mode')
plt.plot(timesteps, wf_predictions.imag, color='gray', linestyle='--', label='prediction 12x12 mode')
plt.plot(timesteps, wavefront.imag - wf_predictions.imag, color='green', label='residual')
plt.legend()
#plt.plot(timesteps, dm_commands, linestyle='-.')
#plt.plot(timesteps, wf_predictions)
plt.show()
