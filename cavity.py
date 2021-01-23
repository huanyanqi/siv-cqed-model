import numpy as np
from siv import SiV
import matplotlib.pyplot as plt

class Cavity:
    """ Model of a cavity with an arbitrary qubit placed in the cavity. The cavity
        has its own resonance frequency while the qubit is assumed to have two 
        resonance frequencies that correspond to its spin state. No model is 
        assumed for the qubit and its resonance frequencies are fixed. """

    # References:
    # An integrated nanophotonic quantum register based on silicon-vacancy spins in diamond, Phys. Rev. B 100, 165428 (2019)
    # Cavity-based quantum networks with single atoms and optical photons, Rev. Mod. Phys.  87, 1379 (2015)
    
    def __init__(self, params=None):
        
        default_params = {
            # Qubit parameters (units: s^-1)
            # Spin down
            "w_down" : 15,       # Spin-down transition frequency
            "g_down" : 5.6,       # Single-photon Rabi frequency
            "gamma_down" : 0.1,   # Transition linewidth / spont. emission rate

            # Spin up
            "w_up" : 17.5,          # Spin-up transition frequency
            "g_up" : 5.6,         # Single-photon Rabi frequency
            "gamma_up" : 0.1,      # Atom linewidth / spont. emission rate

            # Cavity parameters (units: s^-1)
            "w_c" : 0,    # Cavity resonance frequency
            "k_in" : 16.5,    # In-coupling mirror rate
            "k_out" : 0,   # Out-coupling mirror rate
            "k_tot" : 33,   # Cavity linewidth (k_tot = k_in + k_out + k_other)     
        }
        
        # Use the default parameters as a base for inputs that are not provided
        self.cavity_params = default_params.copy()
        if params is not None:
            self.cavity_params.update(params) # Update with user-input params

    def __repr__(self):
        return f"Cavity({str(self.cavity_params)})"
    
    def set_cavity_params(self, new_params):
        """ Update the instance params with a new set of params from a dictionary. """
        self.cavity_params.update(new_params)

    @staticmethod
    def reflectance_function(w, spin_state, w_down, g_down, gamma_down, w_up, g_up, gamma_up, w_c, k_in, k_out, k_tot, **kwargs):
        """ Reflectance as a function of laser frequency w. Used for curve fitting. """
        # From Rev. Mod. Phys.  87, 1379 (2015)

        if spin_state == 0:
            r_down = 1 - (2 * k_in / (1j * (w - w_c) + k_tot + g_down ** 2 / (1j * (w - w_down) + gamma_down)))
            return (r_down * r_down.conjugate()).real
        elif spin_state == 1:
            r_up = 1 - (2 * k_in / (1j * (w - w_c) + k_tot + g_up ** 2 / (1j * (w - w_up) + gamma_up)))
            return (r_up * r_up.conjugate()).real
        elif spin_state == -1:
            r_empty = 1 - (2 * k_in / (1j * (w - w_c) + k_tot))
            return (r_empty * r_empty.conjugate()).real
        else:
            print("spin_state should be -1, 0, or 1.")
            return
               
        # From Christian PRL Fig 2 fitting notebook. Differ by some factors of 2 from the above convention.
        # r_up = 1 - (k_in / (1j * (w - w_c) + (k_tot/2) + g_up ** 2 / (1j * (w - w_up) + (gamma_up/2))))
        # r_down = 1 - (k_in / (1j * (w - w_c) + (k_tot/2) + g_down ** 2 / (1j * (w - w_down) + (gamma_down/2))))
        
        
    @staticmethod
    def transmittance_function(w, spin_state, w_down, g_down, gamma_down, w_up, g_up, gamma_up, w_c, k_in, k_out, k_tot, **kwargs):
        """ Transmittance as a function of laser frequency w. Used for curve fitting. """
        # From Rev. Mod. Phys.  87, 1379 (2015)

        if spin_state == 0:
            t_down = 2 * np.sqrt(k_in * k_out) / (1j * (w - w_c) + k_tot + g_down ** 2 / (1j * (w - w_down) + gamma_down))
            return (t_down * t_down.conjugate()).real
        elif spin_state == 1:
            t_up = 2 * np.sqrt(k_in * k_out) / (1j * (w - w_c) + k_tot + g_up ** 2 / (1j * (w - w_up) + gamma_up))
            return (t_up * t_up.conjugate()).real
        elif spin_state == -1:
            t_empty = 2 * np.sqrt(k_in * k_out) / (1j * (w - w_c) + k_tot)
            return (t_empty * t_empty.conjugate()).real
        else:
            print("spin_state should be -1, 0, or 1.")
            return

    @staticmethod
    def contrast_function(ref_down, ref_up):
        """ Returns the function that defines the contrast between the reflection spectra of down and up spins.
            Accounts for both the ratio of reflectivity as well as the absolute value of the reflectivity. """
        return np.abs(np.log(ref_down / ref_up)) * np.maximum(ref_down, ref_up)
        # (1 - r_down/2)  * r_up # TODO Update this contrast function to infidelity
        
    def reflectance(self, w, spin_state):
        """ Reflectance as a function of laser frequency w. """
        return self.reflectance_function(w, spin_state, **self.cavity_params)
    
    def transmittance(self, w, spin_state):
        """ Transmittance as a function of laser frequency w. """
        return self.transmittance_function(w, spin_state, **self.cavity_params)
    
    def fit_reflection(self, freqs, spectrum):
        """ Fit the reflection spectrum as a function of the laser frequency sweep (freqs). Returns the fitted parameters. """
        raise NotImplementedError
       # def lorentzian(x, A, x0, width):
       #     return A / ((x - x0) ** 2 + width ** 2

class CavitySiV(Cavity):
    """ Model of a cavity with an SiV qubit placed in the cavity. The SiV is 
    modeled using the SiV class and provides the transition frequencies that the 
    cavity experiences. The SiV properties can change based on parameters such as
    the applied B field and strain. """

    def __init__(self, cavity_params=None, siv=None):
        
        # This differs from the generic Cavity with the removal of the 
        # w_up and w_down param here; it will be set by the SiV object later.
        default_cavity_params = {
            # Qubit parameters (units: s^-1)
            # Spin down
            "g_down" : 5.6,       # Single-photon Rabi frequency
            "gamma_down" : 0.1,   # Transition linewidth / spont. emission rate

            # Spin up
            "g_up" : 5.6,         # Single-photon Rabi frequency
            "gamma_up" : 0.1,     # Atom linewidth / spont. emission rate

            # Cavity parameters (units: s^-1)
            "w_c" : 0,      # Cavity resonance frequency
            "k_in" : 16.5,  # In-coupling mirror rate
            "k_out" : 0,    # Out-coupling mirror rate
            "k_tot" : 33    # Cavity linewidth (k_tot = k_in + k_out + k_other)     
        }
        
        # Use the default parameters as a base for inputs that are not provided
        self.cavity_params = default_cavity_params.copy()
        if cavity_params is not None:
            self.cavity_params.update(cavity_params) # Update with user-input params
        
        # Use the default SiV constructor if None is provided
        if siv is None:
            siv = SiV()
        self.siv = siv

        # Set transition frequency from SiV object
        self.cavity_params["w_down"] = 0
        self.cavity_params["w_up"] = self.siv.transition_splitting()

    def __repr__(self):
        return f"CavitySiV({str(self.cavity_params)}, {str(self.siv)})"
    
    def update_siv_params(self, **siv_params):
        """ Update the SiV params with a new set of params as keyword args. """
        self.siv.update_val(**siv_params)
        self.cavity_params["w_up"] = self.siv.transition_splitting()

    def replace_siv(self, siv):
        """ Update the SiV params with a new set of params as keyword args. """
        self.siv = siv
        self.cavity_params["w_up"] = self.siv.transition_splitting()

    # TODO: Make a copy of the cavity + SiV so that we don't need to modify
    # the current parameters? 
    def max_contrast(self, B, delta, B_axis):
        """ Function that we want to optimize over to maximize contrast.
            Will be fed into the optimization routine to find the optimal B and delta. """
        
        B = B * np.array(B_axis)
        w_arr = np.linspace(-15, 15, 1000) # TODO: Make variable, probably depend on detuning?

        # Compute splitting at given field
        self.update_siv_params(B=B)
        splitting = self.siv.transition_splitting() 
        
        # Set the detuning and computed splitting
        self.set_cavity_params({"w_c": delta, "w_down": 0, "w_up": splitting}) 

        contrast = self.contrast_function(self.reflectance(w_arr, 0), self.reflectance(w_arr, 1)) 
        return max(contrast) # Maximum of contrast over the laser frequency spectrum

    # TODO: Make a copy of the cavity + SiV so that we don't need to modify
    # the current parameters? 
    def plot_reflection_contrast(self, w_arr, B, delta):

        # Compute splitting at given field
        self.update_siv_params(B=B)
        splitting = self.siv.transition_splitting()

        # Set the detuning and computed splitting 
        self.set_cavity_params({"w_c": delta, "w_down": 0, "w_up": splitting}) 
        
        ref_down = self.reflectance(w_arr, 0)
        ref_up = self.reflectance(w_arr, 1)
        contrast = self.contrast_function(ref_down, ref_up)
        max_contrast_pos = w_arr[np.argmax(contrast)]
        
        # PLot reflection spectrum
        plt.figure(figsize=[16, 6])
        plt.subplot(1, 2, 1)
        plt.title("Reflection spectrum")
        plt.plot(w_arr, ref_down, label="down")
        plt.plot(w_arr, ref_up, label="up")
        plt.plot([max_contrast_pos, max_contrast_pos], [-0.05, 1], 'r--')
        plt.ylim([0, 1])
        plt.xlabel("Frequency")
        plt.ylabel("Reflectance")
        plt.legend()

        # Plot reflection contrast 
        plt.subplot(1, 2, 2)
        plt.title("Reflection contrast spectrum")
        plt.plot(w_arr, contrast)
        plt.plot([max_contrast_pos, max_contrast_pos] , [min(contrast), max(contrast)], 'r--')
        plt.xlabel("Frequency")
        plt.ylabel("Reflection Contrast")
        
        print("Maximum contrast = {:.3} located at frequency {:.3}".format(max(contrast), max_contrast_pos))
        print("Lower reflectivity = {:.3}, higher reflectivity = {:.3}".format(ref_up[np.argmax(contrast)], ref_down[np.argmax(contrast)]))
