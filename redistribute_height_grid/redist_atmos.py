import h5py

import numpy as np

from helita.sim import rh15d
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import newton

class REDIST_ATMOS:
    """
    Class to redistribute height points in a column from a bifrost atmosphere.
    """

    def __init__(self, ATMOS_FILE):
        """
        Initialises the atmoshpere with ATMOS_FILE as the file-path to the
        bifrost atmoshpere.

        Parameters
        ----------
        ATMOS_FILE : str
            file-path to bifrost atmoshpere
        """
        self.hf = h5py.File(ATMOS_FILE, 'r')

        # If not T_cut
        self.z_max = 0
        
        # Look at first (only) time point
        self.snapshot = 0
        
        # "After-the-fact" tau and height scales from rh
        self.log_tau = None
        self.height = None
        
        # Interpolation order
        self.kind = 'cubic'

    def pick_column(self, idx, idy):
        """
        Picks out a column in the atmosphere file.

        Parameters
        ----------
        idx: int
            x-coordinate of column
        idy: int
            y-coordinate of column
        """

        self.idx = idx
        self.idy = idy

        self.temperature = self.hf['temperature'][self.snapshot][idx][idy]
        self.z = self.hf['z'][self.snapshot]
        self.z_copy = self.z.copy()
        self.temperature_copy = self.temperature.copy()
        
        # interpolation
        self.interpolants = {}
        
        lin_interp = ["B_x", "B_y", "B_z", "velocity_z", "temperature"]
        log_interp = ["electron_density", "density"]
        
        for key in self.hf.keys():
            if key in lin_interp:
                self.interpolants[key] = self._interp_lin(self.hf["z"][self.snapshot],
                                                          self.hf[key][self.snapshot, self.idx, self.idy],
                                                          kind=self.kind)
            elif key in log_interp:
                self.interpolants[key] = self._interp_log(self.hf["z"][self.snapshot],
                                                          self.hf[key][self.snapshot, self.idx, self.idy],
                                                          kind=self.kind)
            elif key == "hydrogen_populations":
                self.interpolants[key] = self._interp_lin(self.hf["z"][self.snapshot],
                                                          self.hf[key][self.snapshot, :, self.idx, self.idy],
                                                          kind=self.kind)
        
    def t_cut(self, T_MAX, max_cut=False, find_max_height=False):
        """
        Method for cutting all arrays at a max height z_max which is defined
        from a temperature limit T_MAX.

        Parameters
        ----------
        T_MAX : float
            Cut the atmosphere at nearest point above T_MAX
        """
        z_max = np.argmin(np.abs(self.temperature - T_MAX))
        
        if self.temperature[z_max] < T_MAX:
            z_max -= 1
            if self.temperature[z_max] < T_MAX:
                print("Warning: temperature does not behave as expected. Column (%d, %d)"
                      %(self.idx, self.idy))
                
        f = interp1d(self.z_copy, self.temperature_copy - T_MAX, kind="cubic")
        z_guess = self.z[z_max]

        self.z0 = newton(f, z_guess)
        self.T0 = f(self.z0) + T_MAX
        
        z = np.linspace(self.z0, self.z.min(), len(self.z[z_max:]))
        
        if find_max_height:
            print(z_max)
        
        if max_cut != False:
            self.z[z_max:] = z
            self.z = self.z[max_cut:]
            
            self.temperature[z_max:] = f(z) + T_MAX
            self.temperature = self.temperature[max_cut:]
            
            z_max = max_cut
        
        else:
            self.z = z
            self.temperature = f(z) + T_MAX
        
        self.z_max = z_max
            
    
    def equispace(self, z=None, temperature=None, N=None, n_tmp=int(1e5)):
        """  
        Function which fills an array z_new such that the points in T and z
        are equidistand placed, i.e. the arc-length between each point is
        constant. Can be called multiple times to refine the redistributed grid.

        Parameters
        ----------
        T : n-D array
            Temperature in K.
        z : n-D array
            Height in m. Grid points to be redistributed
        N : int
            Number of points, defaults to the length z
            
        Returns
        -------
        z_redist : n-D array
            redistributed height grid
        T_redist : n-D array
            temperature at new height points
        """
        if not isinstance(z, np.ndarray):
            if isinstance(temperature, np.ndarray):
                raise ValueError("Bad usage. Both z and temperature should be \
                                  given or neither should be given")
            
            z = self.z
            temperature = self.temperature
            initialise_z = True
        
        else:
            initialise_z = False

        if N == None:
            N = z.shape[0]

        #boost heigth points temporarily
        z_tmp = np.linspace(self.z0, self.z_copy.min(), n_tmp)
        
        #Interpolate
        T_tmp = self.interpolants["temperature"](z_tmp)
        
        #Normalize
        z_norm = np.flip(np.interp(z_tmp,
                                   (z_tmp.min(), z_tmp.max()), 
                                   (0, 1)))
        
        T_norm = np.flip(np.interp(T_tmp,
                                   (T_tmp.min(), T_tmp.max()),
                                   (0, 1)))
        
        #Find arc-length
        arc_length = np.sqrt(1 + np.gradient(T_norm, z_norm)**2)
        
        L = cumtrapz(arc_length, initial=0)
        L0 = L[-1]/(N-1)            
        
        #Redistribute
        z_redist = np.zeros(N)
        T_redist = np.zeros(N)
        T_redist[0] = T_norm[0]
        idz = 1
        for i in range(n_tmp):
            if L[i] > idz*L0:
                z_redist[idz] = z_norm[i]
                T_redist[idz] = T_norm[i]
                idz += 1
                
        z_redist[N-1] = 1
        T_redist[N-1] = T_norm[-1]
    
        #Transform back to original range
        z_redist = np.flip(np.interp(z_redist, 
                                     (0, 1), 
                                     (z_tmp.min(), z_tmp.max())))
        
        T_redist = np.flip(np.interp(T_redist, 
                                     (0, 1), 
                                     (T_tmp.min(), T_tmp.max())))
        
        return z_redist, T_redist

    def _interp_log(self, z, qq, kind='cubic'):
        """
        Linear interpolation converts logarithmically varying quantity to new
        grid. Converts quantity to log-space to perform interpolation.
        
        Parameters
        ----------
        qq : n-D array
            quantity
        z_new : n-D array
            new height grid
        """
        #convert to logspace
        log_q = np.log10(qq)
        
        #interpolate
        f = interp1d(z, log_q, kind=kind)
        
        # Remember to transform back to linear space after interpolation
        # q_new = np.power(10, f(z_new))
        
        return f
    
    def _interp_lin(self, z, qq, kind='cubic'):
        """
        Linear interpolation converts quantity to new grid.
        
        Parameters
        ----------
        qq : n-D array
            quantity
        z : n-D array
            height grid
        """
        #interpolate
        f = interp1d(z, qq, kind=kind)
        
        return f
        
    def make_dict(self, hf_new, z_new, out=True):
        """
        Function to fill a dictionary with redistributed quantities. Interpolates
        to get new values in between height points.
        
        Parameters
        ----------
        hf_new : dict
            empty dictionary to be filled with the redistributed quantities from
            the model atmosphere
        z_new : n-D array
            the redistributed height scale
        out : bool
            print statements
        """
        
        if not isinstance(hf_new, dict):
            raise ValueError("1st argument needs to be a dictionary")
        
        z_shape = len(z_new)
        n_hydr = self.hf["nhydr"].shape[0]
        
        lin_interp = ["B_x", "B_y", "B_z", "velocity_z", "temperature"]
        log_interp = ["electron_density", "density"]
    
        for key in self.hf.keys():
            if out:
                print("--Writing", key, "to dictionary--")
                    
            if key == "z":
                hf_new[key] = np.reshape(z_new, (1, 1, 1, z_shape))
            
            elif key == "hydrogen_populations":
                quant = self.interpolants[key](z_new)
                hf_new[key] = np.reshape(quant, (1, n_hydr, 1, 1, z_shape))
                
            elif key in lin_interp:
                quant = self.interpolants[key](z_new)
                hf_new[key] = np.reshape(quant, (1, 1, 1, z_shape))
            
            elif key in log_interp:
                quant = np.power(10, self.interpolants[key](z_new))
                hf_new[key] = np.reshape(quant, (1, 1, 1, z_shape))
            
            else:
                if out:
                    print("**Skipping", key, "**")
                continue
       
    def nD_dict(self, dictionaries, max_len):
        """
        Make a list of dictionaries to one dictionary. Every data array needs
        to have the same length.
        
        Parameters
        ----------
        dictionaries : list of dicts
            list of dictionaries containing data for atmosphere columns'
        
        max_len : int
            number of data points
            
        Returns
        -------
        nD_dict : dict
            dictionary containing data arrays for multiple columns in
            atmosphere. Useful for creating atmosphere file.
        """
        
        nD_dict = {}
        
        for key in dictionaries[0].keys():
            if key != "hydrogen_populations":
                quant = np.zeros((1, 1, len(dictionaries), max_len))
            else:
                quant = np.zeros((1, 6, 1, len(dictionaries), max_len))
                
            for i in range(len(dictionaries)):
                data = dictionaries[i][key]
                data_shape = data.shape[-1]
                if key != "hydrogen_populations":
                    quant[0, 0, i, :] = data[0, 0, 0, :]             
                
                else:
                    quant[0, :, 0, i, :] = data[0, :, 0, 0, :]
                                 
            nD_dict[key] = quant
        
        return nD_dict
    
    def make_atmos(self, atmos_dict, fname):
        """
        Function to make a hdf5 atmosphere file from a dictionary atmos_dict
        
        Parameters
        ----------
        atmos_dict : dict
            dictionary with the quantities from the model atmosphere
        fname : str
            filename of new atmosphere file
        """
        rh15d.make_xarray_atmos(fname,
                                atmos_dict["temperature"],
                                atmos_dict["velocity_z"],
                                atmos_dict["z"],
                                nH = atmos_dict["hydrogen_populations"],
                                Bz = atmos_dict["B_z"],
                                By = atmos_dict["B_y"], 
                                Bx = atmos_dict["B_x"],
                                rho = atmos_dict["density"],
                                ne = atmos_dict["electron_density"])
    
    def tau_nu(self, data, idx=0, idy=0):
        """
        Obtain tau_nu from data set "data". Rh needs to be run with non-empty
        `ray.input` file.
        
        Parameters
        ----------
        data : hdf5 dataset
            dataset with results from rh
        idx, idy : int
            coordinates of column
            
        Returns
        -------
        tau_nu : n-D array
            optical depth scale
        height : n-D array
            height scale
        """
        height = data.atmos.height_scale[idx, idy].dropna('height')  # first column
        tau_nu = np.zeros((data.ray.chi.dropna('height').shape[2:]))
        for i in range(tau_nu.shape[1]):
            tau_nu[:, i] = cumtrapz(data.ray.chi[0, 0, :, i].dropna('height'),
                                    axis= -1, x=-height,
                                    initial=1e-20)
        
        tau_nu = np.log10(tau_nu)
        self.log_tau = tau_nu
            
        self.height = height

        return tau_nu, height
    
    def tau_redist(self, tau_min, tau_max):
        """
        Method to redistribute height grid wrt. optical depth. Ad hoc method,
        as rh needs to be run to obtain extinction coefficients. Call this method
        after calling method tau_nu with log scale.
        
        Parameters
        ----------
        tau_min : float
            lower limit optical depth to include in height grid
        tau_max : float
            upper limit optical depth to include in height grid
            
        Returns
        -------
        tau_height : n-D array
            redistributed height scale according to optical depth 
        """
        # Cut values outside range 
        tau_cut = self.log_tau*((self.log_tau < tau_max)*(self.log_tau > tau_min))
        
        # Find max difference in tau over wavelengths, and do cumulative sum
        tau_diff = tau_cut[1:, :] - tau_cut[:-1, :]
        tau_sum = np.cumsum(np.max(tau_diff, axis=1))

        # Cut array where sum stagnates --> effectively no height points
        upper_limit = np.where(tau_sum == tau_sum.min())[0][-1]
        lower_limit = np.where(tau_sum == tau_sum.max())[0][0]

        tau_sum = tau_sum[upper_limit:lower_limit]

        tau_height = np.interp(tau_sum,
                              (tau_sum.min(), tau_sum.max()),
                              (self.height[upper_limit],
                               self.height[lower_limit]))
        
        return tau_height