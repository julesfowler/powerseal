## -- This w#ill be where I set up and run actual experiments on the testbed
# -- I think this should also hold all the fiddly bits of how I connect to 
# to hardware or simulation
# ideally easy flag between 

import matplotlib.pyplot as plt
import numpy as np

from hcipy import *


class WavefrontControl:
    """ Right now this is just going to be a dump of a working WFC loop. We'll
    do something smarter/better/faster/stronger later... 
    
    Parameters
    ----------
    wavelength : float
        Science wavelength in meters.
    diameter : float
        Primary mirror diamter in meters.
    mirror_shape : str
        Shape of the mirror. 'circular' or 'segmented' (not yet implemented.)
    n_sample : int
        Pixels per resolution element for the focal grid images.  
    n_airy : int
        Number of airy rings we want when we sample the PSF.
    grid : int
        Pixels across telescope aperture. 
    wfs_type : str
        Wavefront sensor type. 'PyWFS' or ... (not yet implemented.)
    modulation_steps : int
        Steps the modulated pyramid takes. (None for non PyWFS.)
    modulation_radius : float
        Radius of modulation for the pyramid. (None for non PyWFS.) 
    pixels : int
        Number of pixels across the wfs. (Needs to be a factor of grid.)
    wfs_camera : str, HCIPy Camera object
        Either 'noiseless' to spin up a NoiselssDetector or pass an HCIPy
        camera.
    n_actuators : int
        Number of DM actuators. 
    probe : float
        Factor to multiple by wavelength to get probe amplitude. 
    regularization : float
        Strength of the regularlization for the reconstruction matrix. 
    control_frequency : float
        Control frequency for the AO loop in Hz. 
    leak : float
        Leak for a leaky integrator. (1 is no leak.)
    gain : float
        Integrator gain. 
    wfc_control_method : str
        Wavefront control method. 'integrator', 'eof', or 'lmmse' (not yet
        implemented.)
    """


    def __init__(self, wavelength=2.6e-6, diameter=1, mirror_shape='circular', 
                 n_sample=4, n_airy=20, grid=120, wfs_type='PyWFS', 
                 modulation_steps=12, modulation_radius=5, pixels=20,
                 wfs_camera='noiseless', n_actuators=9, probe=.2, 
                 regularization=1e-3, control_frequency=800, leak=.999, 
                 gain=0.3, wfc_control_method='integrator'):
        """ Init function to set up the wavefront control loop. """
        
        self.count = 1
        # Image, sampling, and science parameters
        self.wavelength = wavelength
        self.diameter = diameter
        self.mirror_shape = mirror_shape

        self.n_sample = n_sample
        self.n_airy = n_airy
        self.grid = grid
        self.resolution = self.wavelength/self.diameter
        
        # WFS parameters
        self.wfs_type = wfs_type
        self.modulation = {'steps': modulation_steps, 'radius': modulation_radius}
        self.pixels = pixels
        self.camera_type = wfs_camera
        
        # DM parameters
        self.n_actuators = n_actuators
        self.actuator_spacing = self.diameter/self.n_actuators
        self.probe = probe 
        self.regularization = regularization


        # AO loop parameters 
        self.control_frequency = control_frequency
        self.dt = 1/self.control_frequency
        self.leak = leak
        self.gain = gain
        if wfc_control_method == 'integrator':
            self._update_control = self._update_control_integrator
        else:
            raise NotImplementedError('Only WFC method available at present is integrator.')

        # Define pupil and focal grid and propoagation
        self.pupil_grid = make_pupil_grid(self.grid, self.diameter)
        self.focal_grid = make_focal_grid(q=self.n_sample, num_airy=self.n_airy, spatial_resolution=self.resolution) 
        
        self.propagator = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

    
    def initialize_system(self):
        """ Function to build the optics, atmosphere, and start the wavefront 
        we'll use in a WFC experiment."""
        
        self._build_primary_mirror()
        
        self.wf = Wavefront(self.primary_mirror, wavelength=self.wavelength)
        self.reference_image = self.propagator.forward(self.wf)
        
        self._build_wavefront_sensor()
        self._build_deformable_mirror()
    
        self.initialize_atmosphere()

    def initialize_atmosphere(self, model='single'):
        """ Initialize the atmospheric turbulence."""
       
        def build_multilayer_model():
            #based off of https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf
            heights = np.array([0.0, 2.1, 4.1, 6.5, 9.0, 12.0, 14.8])*1000
            velocities = np.array([6.7, 13.9, 20.8, 29.0, 29.0, 29.0, 29.0])
            outer_scales = np.array([20,20,20,20,20,20,20])
            cn_squared = np.array([0.369, 0.219, 0.127, 0.101, 0.046, 0.111, 0.027])* 1e-12
 
            layers = []
            for h, v, cn, L0 in zip(heights, velocities, Cn_squared,outer_scales):
                layers.append(InfiniteAtmosphericLayer(input_grid, cn, L0, v, h, 2))
            return layers 

        if model == 'single':
            cn_squared = Cn_squared_from_fried_parameter(0.12, self.wavelength)
            layer = InfiniteAtmosphericLayer(self.pupil_grid, cn_squared, 50, 10)
        elif model == 'multilayer':
            layer = MultiLayerAtmosphere(build_multilayer_model(self.pupil_grid))
        else:
            raise NotImplementedError(f"{model} is not currently an implemented turbulence mode.")
        
        self.atmospheric_layer = layer

    def run_wavefront_control(self, n_steps=500, initialize=True):
        """ Workhorse func for the wavefront control. """
    
        # Can I easily output a field to a numpy array/FITS
        # Yes but the physical scale isn't preserved think on this
        
        if initialize:
            
            self.initialize_system()

        self.strehl_record = [] 
        for i in range(n_steps):
            self.count = i
            # update atmosphere layer            
            self.atmospheric_layer.t = i*self.dt
            phase= self.atmospheric_layer.phase_for(self.wavelength) #this is in radians, unwrapped

            self.wf = Wavefront(self.primary_mirror, wavelength=self.wavelength)
            self.wf = self.atmospheric_layer.forward(self.wf)
            # send it through the DM
            #self.deformable_mirror.flatten()
            self.wf = self.deformable_mirror.forward(self.wf)

            # take wfs image
            self._sense_wavefront()
            
            # caculate new DM iteration
            self._update_control()
            
            # move to science image
            self.science_wf = self.propagator.forward(self.wf)
            if self.count%50 == 0:
                plt.clf()
                imshow_field(self.science_wf.intensity)
                plt.show()
                plt.clf()

            # save strehl, phase screen, science image, new DM command, WFS image
            self.strehl = get_strehl_from_focal(self.science_wf.intensity, self.reference_image.intensity)
            self.strehl_record.append(self.strehl)
            print(f'Strehl at iter {i}: {self.strehl}')

        plt.plot(np.arange(0, n_steps), self.strehl_record)
        plt.savefig('strehl.png')
        plt.clf()
        imshow_field(self.science_wf.intensity)
        plt.savefig('final_science.png')
    
    def _build_command_matrix(self):
        """ Function to build a command matrix for the DM. """
        response = []
        modes = self.deformable_mirror.num_actuators 

        for mode in range(modes):
            slope = 0

            for sign in (1, -1):
                amplitude = np.zeros((modes,))
                amplitude[mode] = sign*self.probe*self.wavelength
                self.deformable_mirror.flatten()
                self.deformable_mirror.actuators = amplitude 

                wf_deformable_mirror = self.deformable_mirror.forward(self.wf)
                wf_wavefront_sensor = self.wavefront_sensor.forward(wf_deformable_mirror)
                
                for mod in range(self.modulation['steps']):
                    self.camera.integrate(wf_wavefront_sensor[mod], 1)
                
                response_slopes = self._calculate_wavefront_sensor_slopes(subtract_reference=False, plot=False)
                slope += sign * (response_slopes/2*self.probe*self.wavelength)
            
            response.append(slope.ravel())
        
        modal_response = ModeBasis(response)
        
        self.command_matrix = inverse_tikhonov(modal_response.transformation_matrix, rcond=self.regularization)
        self.deformable_mirror.flatten()

    def _build_deformable_mirror(self):
        """ Function to build a defomrable mirror."""

        influence_functions = make_gaussian_influence_functions(self.pupil_grid, self.n_actuators, self.actuator_spacing)
        deformable_mirror = DeformableMirror(influence_functions)
        
        self.deformable_mirror = deformable_mirror
        self._build_command_matrix()

    def _build_primary_mirror(self):
        """ Function to build a primary mirror."""
        
        if self.mirror_shape == 'circular':
            #primary_mirror = evaluate_supersampled(circular_aperture(self.diameter), self.pupil_grid, self.n_sample)
            primary_mirror = circular_aperture(self.diameter) 
            primary_mirror = primary_mirror(self.pupil_grid)
        
        else:
            raise NotImplementedError("Nothing for that mirror type yet sorry.")
 
        self.primary_mirror = primary_mirror

    def _build_wavefront_sensor(self):
        """ Function to build and initialize a wavefront sensor."""
        
        self.camera = self.camera_type if self.camera_type != 'noiseless'  else NoiselessDetector(self.pupil_grid)
 
        if self.wfs_type == 'PyWFS':
            try:
                pyramid_wfs = PyramidWavefrontSensorOptics(self.pupil_grid, wavelength_0=self.wavelength)
                self.wavefront_sensor = ModulatedPyramidWavefrontSensorOptics(pyramid_wfs, 
                                                                              self.modulation['radius'],
                                                                              self.modulation['steps'])
                       
                self.n_bins = int(self.grid/self.pixels)

                for step in range(self.modulation['steps']):
                    self.camera.integrate(self.wavefront_sensor(self.wf)[step], 1)
                
                self.wfs_reference = self._calculate_wavefront_sensor_slopes(subtract_reference=False)
                self._sense_wavefront = self._sense_wavefront_modulated_pyramid

            except (TypeError, KeyError) as e:
                TypeError("'modulation' must be a dictionary with 'raidus' and 'steps' defined to use PyWFS mode.")
        
        else:
            raise NotImplementedError("Nothing for that WFS type yet sorry.")
        
    def _calculate_wavefront_sensor_slopes(self, subtract_reference=True,
            plot=True):
        """ Calculate slopes from WFS. """
        
        def bin_image(wfs_image, n_bins):
            """ Helper function to bin image. """
            # I feel like this might be built into scipy.griddata? 
            if plot and self.count%50 == 0:
                print('Unbinned image.')
                plt.clf()
                plt.imshow(wfs_image)
                plt.colorbar()
                plt.show()
                plt.clf()
            binned_image = np.zeros((int(wfs_image.shape[0]/n_bins), int(wfs_image.shape[1]/n_bins)))
            for m in np.arange(n_bins-1, wfs_image.shape[0]-n_bins, n_bins):
                for n in np.arange(n_bins-1, wfs_image.shape[1]-n_bins, n_bins):
                    binned_image[int((m+1)/n_bins)-1, int((n+1)/n_bins)-1] = \
                                 np.sum(wfs_image[m-int((n_bins-1)/2):m+int((n_bins-1)/2),
                                                  n-int((n_bins-1)/2):n+int((n_bins-1)/2)])
            
            if plot and self.count%50 == 0:
                print('Binned image.')
                plt.clf()
                plt.imshow(binned_image)
                plt.colorbar()
                plt.show()
                plt.clf()
            return binned_image.flatten()/binned_image.sum()
        
        binned_image = bin_image(self.camera.read_out().shaped, self.n_bins)
        
        pyramid_grid = make_pupil_grid(self.pixels*2, self.diameter)
         
        py_1 = np.array(circular_aperture(self.diameter/2, [-1*self.diameter/4,    self.diameter/4])(pyramid_grid))
        py_2 = np.array(circular_aperture(self.diameter/2, [   self.diameter/4,    self.diameter/4])(pyramid_grid))
        py_3 = np.array(circular_aperture(self.diameter/2, [-1*self.diameter/4, -1*self.diameter/4])(pyramid_grid))
        py_4 = np.array(circular_aperture(self.diameter/2, [   self.diameter/4, -1*self.diameter/4])(pyramid_grid))
        
        normalization = (binned_image[py_1>0]+binned_image[py_2>0]+binned_image[py_3>0]+binned_image[py_4>0])/(4*np.sum(py_1[py_1>0]))

        slopes_x = (binned_image[py_1>0] - binned_image[py_2>0] + binned_image[py_3>0] - binned_image[py_4>0])
        slopes_y = (binned_image[py_1>0] + binned_image[py_2>0] - binned_image[py_3>0] - binned_image[py_4>0])
        
        if subtract_reference:
            self.slopes = (np.array([slopes_x, slopes_y]).flatten() - self.wfs_reference).ravel()
            #self.slopes = (np.array([slopes_x, slopes_y]).flatten()).ravel() #- self.wfs_reference).ravel()
        else:
            self.slopes = (np.array([slopes_x, slopes_y]).flatten()).ravel()
        
        #mask = np.abs(self.slopes) > 0.001
        #self.slopes[mask] = 0

        if plot and self.count%50 == 0:
            D=1
            mid=int(self.slopes.shape[0]/2)
            pyramid_plot_grid = make_pupil_grid(self.pixels, self.diameter) #hardcoded for now/ease
            pyr_mask=circular_aperture(D)(pyramid_plot_grid)
            sx = pyr_mask.copy()
            sy = pyr_mask.copy()
            sx[sx>0]=self.slopes[0:mid]
            sy[sy>0]=self.slopes[mid::]
            sxy=np.asarray([sx, sy]).reshape((2,self.pixels,self.pixels))
            plt.clf()
            plt.imshow(sxy.reshape((2*self.pixels, self.pixels)).transpose())
            plt.colorbar()
            plt.show()
            plt.clf()
        
        mask = np.abs(self.slopes) > 0.001
        self.slopes[mask] = 0

        return self.slopes

    def _sense_wavefront_modulated_pyramid(self):
        """ Function to use the wavefront sensor to take a wavefront image. """ 
        self.sensed_wf = self.wavefront_sensor(self.wf)
        for step in range(self.modulation['steps']):
            self.camera.integrate(self.sensed_wf[step], self.dt/self.modulation['steps'])
        self._calculate_wavefront_sensor_slopes()

    def _update_control_integrator(self):
        """ Basic integrator control method. """

        #self.deformable_mirror.actuators = 
        #self.leak*self.deformable_mirror.actuators.reshape(100,1) - self.gain*self.command_matrix.reshape((100, 632)).dot(self.slopes.reshape((632, 1)))
        #print(np.shape(self.deformable_mirror.actuators))
        #self.deformable_mirror.actuators = (self.leak*self.deformable_mirror.actuators.reshape(self.n_actuators**2,1) - self.gain*self.command_matrix.reshape((self.n_actuators**2, 632)).dot(self.slopes.reshape((632, 1)))).ravel()
        self.deformable_mirror.actuators = (self.leak*self.deformable_mirror.actuators.reshape(self.n_actuators**2,1) - self.gain*self.command_matrix.dot(self.slopes.reshape((632, 1)))).ravel()
        #self.deformable_mirror.actuators = (self.leak*self.deformable_mirror.actuators.reshape(self.n_actuators**2,1) - self.gain*self.command_matrix.reshape((self.n_actuators**2, 632)).dot(np.zeros_like(self.slopes.reshape((632, 1))))).ravel()
        #print(np.shape(self.gain*self.command_matrix.reshape((100, 632)).dot(self.slopes.reshape((632, 1)))))

if __name__ == "__main__":
    wfc = WavefrontControl()
    wfc.run_wavefront_control()
