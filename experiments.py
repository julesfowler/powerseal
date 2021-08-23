## -- This will be where I set up and run actual experiments on the testbed
# -- I think this should also hold all the fiddly bits of how I connect to 
# to hardware or simulation
# ideally easy flag between 

from hcipy import *


class WavefrontControl:
    """ Right now this is just going to be a dump of a working WFC loop. We'll
    do something smarter/better/faster/stronger later... """

    def __init__(wavelength, diameter, n_sample, n_airy, grid, control_frequency, 
                 leak, gain, wfc_control_method):
        
        # Image, sampling, and science parameters
        self.wavelength = wavelength
        self.diameter = diameter

        self.n_sample = n_sample
        self.n_airy = n_airy
        self.grid = grid
        self.resolution = self.wavelength/self.diameter
        
        # DM parameters
        self.n_actuators = n_actuators
        self.probe = probe
        self.regularlization = regularization


        # AO loop parameters 
        self.control_frequency = control_frequency
        self.dt = 1/self.control_frequency
        self.leak = leak
        self.gain = gain
        if wfc_control_method == 'integrator':
            self._update_control = _update_control_integrator
        else:
            raise NotImplementedError('Only WFC method available at present is integrator.')

        # Define pupil and focal grid and propoagation
        self.pupil_grid = make_pupil_grid(self.grid, self.diameter)
        self.focal_grid = make_focal_grid(q=self.n_sample, num_airy=self.n_airy, spatial_resolution=self.resolution) 
        
        self.propagator = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

    
    def build_optics(self):
        
        self._build_primary_mirror()
        self._build_wavefront_sensor()
        self._build_deformable_mirror()
    
    def initialize_atmosphere(self):
        """ Initialize the atmosphere, but realistically it needs to evolve with
        each DM iter, so this can't just spit something out. """
       
        # FIXME: do we want an option for a simpler model tooooo?
        # Probably 
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

        layer = MultiLayerAtmosphere(build_multilayer_model(self.pupil_grid))
        self.atmospheric_layer = layer

    def run_wavefront_control(self, n_steps, initialize=True):
        """ Workhorse func for the wavefront control. """
    
        # Can I easily output a field to a numpy array/FITS
        # Yes but the physical scale isn't preserved think on this
        
        if initialize:
            
            # start with a clean wf
            self.wf = Wavefront(self.primary_mirror, wavelength=self.wavelength)
            self.reference_image = self.propagator.forward(self.wf)
            
            # build optics and initialize atmosphere
            # Note that we have to make the wf first in case we need to make a WFS reference image
            self.build_optics()
            self.initialize_atmosphere()

        for i in range(n_steps):
            
            # update atmosphere layer            
            self.layer.t = i*self.dt
            phase= layer.phase_for(self.wavelength) #this is in radians, unwrapped
            self.wf = self.layer.forward((self.wf))
            
            # send it through the DM
            self.wf = self.deformable_mirror.forward(self.wf)

            # take wfs image
            self._sense_wavefront()
            
            # caculate new DM iteration
            self._update_control()
            
            # move to science image
            self.wf = self.propagator.forward(self.wf)

            # save strehl, phase screen, science image, new DM command, WFS image
            self.strehl = get_strehl_from_focal(self.wf.intensity, self.reference_image.intensity)
            print(f'Strehl at iter {i}: {self.strehl}')


    def _sense_wavefront_modulated_pyramid(self):
        
        self.sensed_wf = self.wavefront_sensor(self.wf)
        for step in range(self.modulation['steps']):
            self.camera.integrate(self.sense_wf[step], self.dt/self.modulation['steps'])
        self._calculate_wavefront_sensor_slopes()


    def _calculate_wavefront_sensor_slopes(self):
            
        """ Calculate slopes from WFS. """
        
        def bin_image(wfs_image, n_bins):
            """ Helper function to bin image. """
            # I feel like this might be built into scipy.griddata? 
            binned_image = np.zeros((int(wfs_image.shape[0]/n_bins), int(wfs_image.shape[1]/n_bins)))
            for m in np.arange(n_bins-1, wfs_image.shape[0]-n_bins, n_bins):
                for n in np.arange(n_bins-1, wfs_image.shape[1]-n_bins, n_bins):
                    binned_image[int((m+1)/n_bins)-1, int((n+1)/n_bins)-1] = \
                                 np.sum(wfs_image[m-int((n_bins-1)/2):m+int((n_bins-1)/2),
                                                  n-int((n_bins-1)/2):n+int((n_bins-1)/2)])
            return binned_image.flatten()/binned_image.sum()
        
        binned_image = (self.camera.read_out().shaped, self.n_bins)
        
        pyramid_grid = make_pupil_grid(self.pixels*2, self.diameter)
         
        py_1 = circular_aperture(self.diameter/2, [-1*self.diameter/4,    self.diameter/4])(pyramid_grid)
        py_2 = circular_aperture(self.diameter/2, [   self.diameter/4,    self.diameter/4])(pyramid_grid)
        py_3 = circular_aperture(self.diameter/2, [-1*self.diameter/4, -1*self.diameter/4])(pyramid_grid)
        py_4 = circular_aperture(self.diameter/2, [   self.diameter/4, -1*self.diameter/4])(pyramid_grid)
        
        normalization = (binned_image[py_1>0]+image[py_2>0]+image[py_3>0]+image[py_4>0])/(4*np.sum(py1[py_1>0]))

        slopes_x = (binned_image[py_1>0] - image[py_2>0] + image[py_3>0] - image[py_4>0])
        slopes_y = (binned_image[py_1>0] + image[py_2>0] - image[py_3>0] - image[py_4>0])
        
        self.slopes = (np.array([slopes_x, slopes_y]).flatten() - self.wfs_reference).ravel()
        return self.slopes

    def _update_control_integrator(self):
        
        """ Basic integrator control method. """
        self.deformable_mirror.actuators = self.leak*deformable_mirror.actuators - self.gain*self.command_matrix.dot(self.slopes)

    def _build_wavefront_sensor(self, modulation=None, camera=None, wfs='PyWFS'):
        
        self.camera = camera if camera is not None else NoiselessDetector(self.pupil_grid)
 
        if wfs == 'PyWFS':
            try:
                pyramid_wfs = PyramidWavefrontSensorOptics(self.pupil_grid, wavelength_0=self.wavelength)
                wavefront_sensor = ModulatedPyramidWavefrontSensorOptics(pyramid_wfs, 
                                                                              modulation['radius'],
                                                                              modulation['steps'])
                
                self.n_bins = int(self.grid_size/self.pixels)

                for step in range(modulation['steps']):
                    self.camera.integrate(wavefront_sensor(self.wf)[step], 1)
                
                self.wfs_reference = self._calculate_wavefront_sensor_slopes()
                self._sense_wavefront = self._sense_wavefront_modulated_pyramid

            except (TypeError, KeyError) as e:
                TypeError("'modulation' must be a dictionary with 'raidus' and 'steps' defined to use PyWFS mode.")
        
        else:
            raise NotImplementedError("Nothing for that WFS type yet sorry.")
        
    def _build_primary_mirror(self, mirror_shape='circular', size=1, n_samples=6):
        if mirror_shape == 'circular':
            primary_mirror = evaluate_supersampled(circular_aperture(size), self.pupil_grid, n_samples)
        
        else:
            raise NotImplementedError("Nothing for that mirror type yet sorry.")
 
        self.primary_mirror = primary_mirror

    def _build_deformable_mirror(self, n_actuators=32, actuator_spacing=1/32):

        influence_functions = make_gaussian_influence_functions(self.pupil_grid, n_actuators, actuator_spacing)
        deformable_mirror = DeformableMirror(influence_functions)
        
        self.deformable_mirror = deformable_mirror
        self._build_command_matrix()

    def _build_command_matrix(self):
        
        response = []
        modes = self.deformable_mirror.num_actuators 

        for mode in range(modes):
            slope = 0

            for sign in (1, -1):
                amplitude = np.zeros((modes,))
                amplitude[mode] = sign*self.probe*self.wavelength
                deformable_mirror.flatten()
                deformable_mirror.actuators = amplitude 

                wf_deformable_mirror = self.deformable_mirror.forward(self.wf)
                wf_wavefront_sensor = self.wavefront_Sensor.forward(wf_deformable_mirror)
                
                for mod in range(self.modulation['steps']):
                    self.camera.integrate(wf_wavefront_sensor[mod], 1)
                
                response_slopes = self._calculate_wavefront_sensor_slopes()
                slope += sign * (response_slopes/2*self.probe*self.wavelength)
            
            response.append(slope.ravel())
        
        modal_response = ModeBasis(response)
        
        self.command_matrix = inverse_tikhonov(modal_response, rcond=self.regularization)

