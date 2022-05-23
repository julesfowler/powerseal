## -- IMPORTS
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import numpy as np
import scipy


## -- FUNCTIONS

def build_complex_basis(i0, j0, m0, n0):
    A = np.zeros((m0, n0,i0*j0), dtype=complex)
    c = 0
    for i in range(int(-1*i0/2+1), int(i0/2+1)):
        for j in range(int(-1*j0/2+1), int(j0/2+1)):
            for m in range(m0):
                for n in range(n0):
                    A[m, n, c] = complex(np.cos((i*m + j*n)*2*np.pi/m0), np.sin((i*m + j*n)*2*np.pi/m0))
            c+=1
    A_reshape = A.reshape(m0*n0, i0*j0)

    return A_reshape


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


def build_complex_alphas(alphas_real, alphas_imag, time_interval=2000):

    complex_alphas = []
    for index, elem in enumerate(alphas_real):
        phi = 2*np.pi*alphas_imag[index]/time_interval
        exp_term = complex(0,1)*phi
        complex_alpha = elem*np.exp(exp_term)
        complex_alphas.append(complex_alpha)

    return complex_alphas


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


def build_K(P_s, C, P_v, l):
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

def convert_phase_to_wfe(phase, wavelength=500e-9, unit_conversion=1e9):
    
    wfe = phase*(wavelength*unit_conversion)/(2*np.pi)
    
    return wfe 

if __name__ == "__main__":
    
    ## -- GET IT
    i0, j0, m0, n0 = 32, 32, 32, 32
    n_sample = 9998
    fourier_modes = build_complex_basis(i0, j0, m0, n0)
    print('Built fourier modes.')
    for n_frames in [10000]:
        infile = 'modes=32_frames=10000_poyneer_1_franken_2khz_infinite_48x_future.fits'
        hdu = fits.open(infile)
        modes = np.complex(0, 1)*hdu[0].data + np.complex(1, 0)*hdu[1].data
        """
        #wavefront = modes[1715, :]


        # starting with 4 wind layers at mode 12 x 12
        #l = 4
        #sigma_squared = [7.1216722414696945e-06,
        #                 6.541757149605491e-06,
        #                 4.870870115743126e-06,
        #                 4.238942978009774e-06,
        #                 6.1543419962556235e-06]
        #alpha = [0.995823704690824,
        #         0.9960367335962748,
        #         0.9986817173944081,
        #         0.9986202334957323,
        #         0.9969061483950212]

        # perfect DM for now, so d[t] = x[t]
        dt_start = 4
        sample_length = 10000
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
        K = build_K(P_s, C, P_v, l)

        # calculate initial state vector 
        #w_layers = np.zeros(l+1) # white noise set to zero for now
        w_layers = np.random.random(l+1)*np.complex(np.std(wavefront.real), np.std(wavefront.imag))
        a_init = w_layers #build_autoregression(alpha, w_layers, wavefront[0])
        x = build_x_init(a_init, wavefront, dm_commands, dt_start)

        wf_predictions = np.zeros_like(wavefront)
        wf_integrator = wavefront[1:]
        for dt in timesteps:
            
            y = measurement_data[dt]
            dm_prev = dm_commands[dt-1]
            x, wf_prediction = evolve_state(x, dm_prev, y, A, C, G, K)
            #if dt < 28000:
                #print(wf_prediction, wavefront[dt+1])
            #dm_command = calculate_dm_command(wf_prediction)

            wf_predictions[dt] = wf_prediction
            #dm_commands[dt] = dm_command

        #plt.plot(timesteps, measurement_data.real)
        plt.title('Imaginary Coefficient')
        plt.semilogy(timesteps, np.abs(wavefront.imag), color='gray', label='simulated data 12x12 mode')
        #plt.semilogy(timesteps, wf_predictions.real, color='gray', linestyle='--', label='prediction 12x12 mode')
        plt.semilogy(timesteps[:-1], np.abs(wavefront[:-1].imag - wf_integrator.imag), color='green', label='residual integrator')
        plt.semilogy(timesteps, np.abs(wavefront.imag - wf_predictions.imag), color='cyan', label='residual predictor')
        plt.legend()
        #plt.plot(timesteps, dm_commands, linestyle='-.')
        #plt.plot(timesteps, wf_predictions)
        plt.show()
        """
        """
        infile = f'mode_layers_training={n_frames}.csv'
        print(f'For {n_frames} of training data:')
        layer_data = ascii.read(infile)
        index_data = ascii.read('mode_dict_32x.csv')
        prediction_out = {}
        
        for index in range(1024):
            print(index)
            sample_length = 10000
            wavefront = modes[index][:sample_length]
            mode = index_data['k'][index], index_data['l'][index]

            alphas_real = layer_data[f'{str(index)}_alphas_real']
            alphas_imag = layer_data[f'{str(index)}_alphas_imag']
            alpha = build_complex_alphas(alphas_real[alphas_real > 0], alphas_imag[alphas_real > 0])
            sigmas = layer_data[f'{str(index)}_sigmas']
            sigma_squared = sigmas[alphas_real > 0]

            l = len(sigma_squared) - 1
            
            # perfect DM for now, so d[t] = x[t]
            dt_start = 4
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
            K = build_K(P_s, C, P_v, l)

            # calculate initial state vector 
            #w_layers = np.zeros(l+1) # white noise set to zero for now
            w_layers = np.random.random(l+1)*np.complex(np.std(wavefront.real), np.std(wavefront.imag))
            a_init = w_layers #build_autoregression(alpha, w_layers, wavefront[0])
            x = build_x_init(a_init, wavefront, dm_commands, dt_start)

            wf_predictions = np.zeros_like(wavefront)
            wf_integrator = wavefront[2:]
            for dt in timesteps:
            
                y = measurement_data[dt]
                dm_prev = dm_commands[dt-1]
                x, wf_prediction = evolve_state(x, dm_prev, y, A, C, G, K)

                wf_predictions[dt] = wf_prediction
            
            
            #plt.title('Imaginary Coefficient')
            #plt.semilogy(timesteps, np.abs(wavefront.imag), color='gray', label=f'simulated data {str(mode)} mode')
            prediction_out[f'wavefront_imag_{index}'] = wavefront.imag[:-2]
            prediction_out[f'wavefront_real_{index}'] = wavefront.real[:-2]
            prediction_out[f'prediction_real_{index}'] = wf_predictions.real[:-2]
            prediction_out[f'prediction_imag_{index}'] = wf_predictions.imag[:-2]
            prediction_out[f'integrator_real_{index}'] = wf_integrator.real
            prediction_out[f'integrator_imag_{index}'] = wf_integrator.imag
            #plt.semilogy(timesteps[:-1], np.abs(wavefront[:-1].imag - wf_integrator.imag), color='green', label='residual integrator')
            #plt.semilogy(timesteps, np.abs(wavefront.imag - wf_predictions.imag), color='cyan', label='residual predictor')
            #plt.legend()
            #plt.plot(timesteps, dm_commands, linestyle='-.')
            #plt.plot(timesteps, wf_predictions)
            #plt.savefig(f'imag_coeff_{str(mode)}.png')
            #plt.clf()
            #plt.title('Real Coefficient')
            #plt.semilogy(timesteps, np.abs(wavefront.real), color='gray', label=f'simulated data {str(mode)} mode')
            #plt.semilogy(timesteps[:-1], np.abs(wavefront[:-1].real - wf_integrator.real), color='green', label='residual integrator')
            #plt.semilogy(timesteps, np.abs(wavefront.real - wf_predictions.real), color='cyan', label='residual predictor')
            #plt.legend()
            #plt.savefig(f'real_coeff_{str(mode)}.png')
            #plt.clf()

        ascii.write(prediction_out, f'prediction_out_single_training={n_frames}_real.csv', overwrite=True)
        """

        
        prediction = ascii.read(f'prediction_out_single_training={n_frames}_real.csv')


        n_modes = np.shape(fourier_modes)[0]
        wavefront_coeffs = np.zeros((n_modes, n_sample), dtype=complex)
        prediction_coeffs = np.zeros((n_modes, n_sample), dtype=complex)
        integrator_coeffs = np.zeros((n_modes, n_sample), dtype=complex)
        
        for index in range(n_modes):
            print(index)

            wavefront_coeffs[index, :] = prediction[f'wavefront_real_{index}'] + np.complex(0, 1)*prediction[f'wavefront_imag_{index}']
            prediction_coeffs[index, :] = prediction[f'prediction_real_{index}'] + np.complex(0, 1)*prediction[f'prediction_imag_{index}']
            #integrator_coeffs[index, :] = prediction[f'integrator_real_{index}'] + np.complex(0, 1)*prediction[f'integrator_imag_{index}']
            

        wavefront = np.matmul(fourier_modes, wavefront_coeffs)
        delta_t = 1
        wf_integrator_residuals = np.zeros_like(wavefront.real)
        dt_integrator = np.arange(delta_t, 9998-delta_t)
        for i in range(delta_t, 9998-delta_t):
            wf_integrator_residuals[:, i] = wavefront.real[:, i+delta_t] - wavefront.real[:, i]
        wf_integrator_residuals = wf_integrator_residuals[:, delta_t:9998-delta_t]
        print(np.shape(wf_integrator_residuals), np.shape(dt_integrator))
        wf_prediction = np.matmul(fourier_modes, prediction_coeffs)
        #wf_integrator = np.matmul(fourier_modes, integrator_coeffs)
        #wf_integrator = np.zeros_like(wavefront.real)
        wind_prediction_hdu = fits.HDUList([fits.PrimaryHDU(wavefront.real), fits.ImageHDU(wavefront.imag), 
        #                                    fits.ImageHDU(wf_integrator.real), fits.ImageHDU(wf_integrator.imag),
                                            fits.ImageHDU(wf_prediction.real), fits.ImageHDU(wf_prediction.imag)])

        wind_prediction_hdu.writeto(f'wind_prediction_{n_frames}_complex_out.fits', overwrite=True)
        plt.clf()
        dt = np.arange(n_sample)
        plt.semilogy(dt, np.sqrt(np.mean(convert_phase_to_wfe(wavefront.real)**2, axis=0)), color='gray', label='uncorrected')
        plt.semilogy(dt_integrator, np.sqrt(np.mean(convert_phase_to_wfe(wf_integrator_residuals)**2, axis=0)), color='green', label='quasi integrator')
        plt.semilogy(dt, np.sqrt(np.mean(convert_phase_to_wfe(wavefront.real - wf_prediction.real)**2, axis=0)), color='cyan', label='prediction')
        plt.legend()
        med_turb = np.median(np.sqrt(np.mean(convert_phase_to_wfe(wavefront.real)**2, axis=0)))
        med_integrator = np.median(np.sqrt(np.mean(convert_phase_to_wfe(wf_integrator_residuals)**2, axis=0)))
        med_predictor = np.median(np.sqrt(np.mean(convert_phase_to_wfe(wavefront.real - wf_prediction.real)**2, axis=0)))
        plt.xlabel('Iterations [2kHz]')
        plt.ylabel('RMS wavefront error [nm]')
        plt.title(f'PFC RMS = {med_predictor} nm')
        plt.ylim(1, 1e3)
        print(med_turb, med_integrator, med_predictor)
        plt.show()
        
