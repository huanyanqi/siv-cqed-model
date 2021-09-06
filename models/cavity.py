import copy
import numpy as np
from sivqed.models.siv import SiV
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MultiQubitCavity:
    """ Model of a cavity with any number of arbitrary qubits placed in the 
        cavity. The cavity has its own resonance frequency while the qubit is 
        assumed to have two  resonance frequencies that correspond to its spin 
        state. No model is  assumed for the qubit and its resonance frequencies 
        are fixed. 

        References:
        # An integrated nanophotonic quantum register based on silicon-vacancy spins in diamond, Phys. Rev. B 100, 165428 (2019)
        # Cavity-based quantum networks with single atoms and optical photons, Rev. Mod. Phys.  87, 1379 (2015)
    """

    default_cavity_params = {
            # Cavity parameters (units: s^-1)
            "w_c" : 0,    # Cavity resonance frequency
            "k_in" : 16.5,    # In-coupling mirror rate
            "k_out" : 0,   # Out-coupling mirror rate
            "k_tot" : 33,   # Cavity linewidth (k_tot = k_in + k_out + k_other)     
        }

    default_qubit_params = {
        # Qubit parameters (units: s^-1)
        # Spin down
        "w_down" : 15,       # Spin-down transition frequency
        "g_down" : 5.6,       # Single-photon Rabi frequency
        "gamma_down" : 0.1,   # Transition linewidth / spont. emission rate

        # Spin up
        "w_up" : 17.5,          # Spin-up transition frequency
        "g_up" : 5.6,         # Single-photon Rabi frequency
        "gamma_up" : 0.1,      # Atom linewidth / spont. emission rate
    }
    
    def __init__(self, cavity_params=None, qubit_params=None):
        
        # Use the default parameters as a base then update with user-input params
        self.cavity_params = self.default_cavity_params.copy()
        if cavity_params is not None:
            self.cavity_params.update(cavity_params)
            
        self.qubit_params = [] 

         # Create a default qubit for each qubit specified, then update with 
         # provided values
        if qubit_params is not None:
            if type(qubit_params) != list:
                qubit_params = list(qubit_params)

            for params in qubit_params:
                self.qubit_params.append(self.default_qubit_params.copy())
                self.qubit_params[-1].update(params)  
        
        # If nothing provided, default is a single qubit in a 1-element list
        else:
            self.qubit_params = [self.default_qubit_params.copy()] 

    def __repr__(self):
        return f"MultiQubitCavity({str(self.cavity_params)}, {str(self.qubit_params)})"

    def set_cavity_params(self, cavity_params):
        """ Update the instance params with a new set of params from a dictionary. """
        self.cavity_params.update(cavity_params)

    def set_qubit_params(self, qubit_params):
        """ Update the instance params with a new set of params from a dictionary. 
        
        qubit_params : dict
            Key: Index of the qubit to be updated
            Value: Dictionary of qubit parameters to be updated.
        """
        for idx, params in qubit_params.items():
            if idx >= len(self.qubit_params):
                print(f"Provided qubit index of {idx} exeeded number of qubits in system!")
                continue
            self.qubit_params[idx].update(params)

    @classmethod
    def complex_reflect_coeff_fn(cls, w, spin_state, qubit_params, w_c, k_in, k_tot, **kwargs):
        """ Complex reflection coefficient as a function of laser frequency w.
            Reflectance (power) is the mod square of this coefficient. """

        # From Rev. Mod. Phys.  87, 1379 (2015)

        denom = 1j * (w - w_c) + k_tot 

        if spin_state == 0:
            for params in qubit_params:
                denom += params["g_down"] ** 2 / (1j * (w - params["w_down"]) + params["gamma_down"])
        elif spin_state == 1:
            for params in qubit_params:
                denom += params["g_up"] ** 2 / (1j * (w - params["w_up"]) + params["gamma_up"])
        elif spin_state == -1:
            pass
        else:
            print("spin_state should be -1, 0, or 1.")
            return
        
        r = 1 - (2 * k_in / denom)
        return r

    @classmethod
    def reflectance_fn(cls, w, spin_state, qubit_params, w_c, k_in, k_tot, **kwargs):
        """ Reflectance as a function of laser frequency w. Taken to be |r(w)|^2 where 
            r is the complex reflection coefficient. """
        # From Rev. Mod. Phys.  87, 1379 (2015)

        # From Christian PRL Fig 2 fitting notebook. Differ by some factors of 2 from the above convention.
        # r_up = 1 - (k_in / (1j * (w - w_c) + (k_tot/2) + g_up ** 2 / (1j * (w - w_up) + (gamma_up/2))))
        # r_down = 1 - (k_in / (1j * (w - w_c) + (k_tot/2) + g_down ** 2 / (1j * (w - w_down) + (gamma_down/2))))  
        
        r = cls.complex_reflect_coeff_fn(w, spin_state, qubit_params, w_c, k_in, k_tot, **kwargs)
        return (r * r.conjugate()).real

    @classmethod
    def complex_transmit_coeff_fn(cls, w, spin_state, qubit_params, w_c, k_in, k_out, k_tot, **kwargs):
        """ Complex transmission coefficient as a function of laser frequency w.
            Transmittance (power) is the mod square of this coefficient. """

        # From Rev. Mod. Phys.  87, 1379 (2015)

        denom = 1j * (w - w_c) + k_tot

        if spin_state == 0:
            for params in qubit_params:
                denom += params["g_down"] ** 2 / (1j * (w - params["w_down"]) + params["gamma_down"])
        elif spin_state == 1:
            for params in qubit_params:
                denom += params["g_up"] ** 2 / (1j * (w - params["w_up"]) + params["gamma_up"])
        elif spin_state == -1:
            pass
        else:
            print("spin_state should be -1, 0, or 1.")
            return

        t = 2 * np.sqrt(k_in * k_out) / denom
        return t

    @classmethod
    def transmittance_fn(cls, w, spin_state, qubit_params, w_c, k_in, k_out, k_tot, **kwargs):
        """ Transmittance as a function of laser frequency w. Taken to be |t(w)|^2 where 
            t is the complex transmission coefficient. """
        # From Rev. Mod. Phys.  87, 1379 (2015)

        t = cls.complex_transmit_coeff_fn(w, spin_state, qubit_params, w_c, k_in, k_out, k_tot, **kwargs)
        return (t * t.conjugate()).real

    def reflectance(self, w, spin_state):
        """ Reflectance as a function of laser frequency w. """
        return self.reflectance_fn(w, spin_state, self.qubit_params, **self.cavity_params)

    def reflected_phase(self, w, spin_state):
        """ Reflected phase as a function of laser frequency w. """
        return np.angle(self.complex_reflect_coeff_fn(w, spin_state, self.qubit_params, **self.cavity_params))
    
    def transmittance(self, w, spin_state):
        """ Transmittance as a function of laser frequency w. """
        return self.transmittance_fn(w, spin_state, self.qubit_params, **self.cavity_params)

    def trasnmitted_phase(self, w, spin_state):
        """ Transmitted phase as a function of laser frequency w. """
        return np.angle(self.complex_transmit_coeff_fn(w, spin_state, self.qubit_params, **self.cavity_params))
        

class Cavity(MultiQubitCavity):
    """ Model of a cavity with an arbitrary qubit placed in the cavity. The cavity
        has its own resonance frequency while the qubit is assumed to have two 
        resonance frequencies that correspond to its spin state. No model is 
        assumed for the qubit and its resonance frequencies are fixed.

        References:
        # An integrated nanophotonic quantum register based on silicon-vacancy spins in diamond, Phys. Rev. B 100, 165428 (2019)
        # Cavity-based quantum networks with single atoms and optical photons, Rev. Mod. Phys.  87, 1379 (2015)
    """
    
    ### Initialization ###

    def __init__(self, cavity_params=None, qubit_params=None):
        
        # Use the default parameters as a base then update with user-input params
        self.cavity_params = self.default_cavity_params.copy()
        if cavity_params is not None:
            self.cavity_params.update(cavity_params)

        self.qubit_params = self.default_qubit_params.copy()
        if qubit_params is not None:
            self.qubit_params.update(qubit_params)  
    
    def __repr__(self):
        return f"Cavity({str(self.cavity_params)}, {str(self.qubit_params)})"
    
    def set_qubit_params(self, qubit_params):
        """ Update the instance params with a new set of params from a dictionary. """
        self.qubit_params.update(qubit_params)

    ### Reflectance and Transmittance ###
    
    def reflectance(self, w, spin_state):
        """ Reflectance as a function of laser frequency w. """
        return self.reflectance_fn(w, spin_state, [self.qubit_params], **self.cavity_params)

    def reflected_phase(self, w, spin_state):
        """ Reflected phase as a function of laser frequency w. """
        return np.angle(self.complex_reflect_coeff_fn(w, spin_state, [self.qubit_params], **self.cavity_params))
    
    def transmittance(self, w, spin_state):
        """ Transmittance as a function of laser frequency w. """
        return self.transmittance_fn(w, spin_state, [self.qubit_params], **self.cavity_params)

    def trasnmitted_phase(self, w, spin_state):
        """ Transmitted phase as a function of laser frequency w. """
        return np.angle(self.complex_transmit_coeff_fn(w, spin_state, [self.qubit_params], **self.cavity_params))

    ### Spin Contrast - not defined for multi-qubit ###

    @staticmethod
    def spin_contrast_fn(ref_down, ref_up):
        """ Function that defines the contrast between the reflection
        spectra of down and up spins that is used during optimization.  """
        
        return np.abs(np.log(ref_down / ref_up)) * np.maximum(ref_down, ref_up)
        # (1 - r_down/2)  * r_up # TODO Update this contrast function to infidelity?

    @staticmethod
    def spin_true_contrast_fn(ref_down, ref_up):
        """ Returns the actual contrast between the reflection
        spectra of down and up spins. Not used for optimization since it does
        not fully represent all our optimization goals.  """

        return np.maximum(ref_down, ref_up) / np.minimum(ref_down, ref_up)

    @staticmethod
    def empty_contrast_fn(ref_empty, ref_up):
        """ Function that defines the contrast between the reflection
        spectra with an up spin and with an empty cavity.  """
        return np.abs(np.log(ref_empty / ref_up))

    def empty_contrast(self, w, w_up):
        """ Function that we want to optimize over to maximize contrast.
            Will be fed into the optimization routine to find the optimal B and delta. """

        c = copy.deepcopy(self)
        c.set_cavity_params({"w_c": 0})
        c.set_qubit_params({"w_up": w_up}) 

        return c.empty_contrast_fn(c.reflectance(w, -1), c.reflectance(w, 1)) 

    def optimize_empty_contrast(self, w_0, w_up_0, w_bounds, w_up_bounds):
        opt = minimize(lambda args: -self.empty_contrast(*args), x0=[w_0, w_up_0], bounds=(w_bounds, w_up_bounds))

        # Extract optimal params and optimal value
        w_opt, w_up_opt = opt.x 
        empty_contrast_opt = -opt.fun # Negative since we used minimize()

        return ((w_opt, w_up_opt), empty_contrast_opt)
    
    def spin_contrast_no_SiVModel(self, w, delta, splitting):
        """ Function that we want to optimize over to maximize contrast.
            Does not assume an SiV model and thus we provide the qubit splitting instead of B field.
            Will be fed into the optimization routine to find the optimal delta. """
        
        c = copy.deepcopy(self)

        # Set the input detuning and splitting
        c.set_cavity_params({"w_c": delta})
        c.set_qubit_params({"w_down": 0, "w_up": splitting}) 

        return c.spin_contrast_fn(c.reflectance(w, 0), c.reflectance(w, 1)) 
    
    def optimize_spin_contrast_no_SiVmodel(self, w_0, delta_0, splitting_0, 
                                                 w_bounds, delta_bounds, splitting_bounds):
        """ Optimize the contrast between the two spin states of the qubit but without
        assuming any SiV model, and thus B field does not come into play. Instead we 
        impose a maximum splitting allowed for the two states. """        

        opt = minimize(lambda args: -self.spin_contrast_no_SiVModel(args[0], args[1], args[2]), 
                        x0=[w_0, delta_0, splitting_0], bounds=(w_bounds, delta_bounds, splitting_bounds))

        # Extract optimal params and optimal value
        w_opt, delta_opt, splitting_opt = opt.x
        spin_contrast_opt = -opt.fun # Negative since we used minimize()

        return ((w_opt, delta_opt, splitting_opt), spin_contrast_opt)
    
    def plot_reflection_contrast_no_SiVmodel(self, w_arr, delta, splitting, print_output=True):

        c = copy.deepcopy(self)
        
        # Set the detuning and computed splitting 
        c.set_cavity_params({"w_c": delta})
        c.set_qubit_params({"w_down": 0, "w_up": splitting}) 
        
        ref_down = c.reflectance(w_arr, 0)
        ref_up = c.reflectance(w_arr, 1)
        opt_value_fn = c.spin_contrast_fn(ref_down, ref_up)
        max_optval_pos = w_arr[np.argmax(opt_value_fn)]
        contrast = c.spin_true_contrast_fn(ref_down, ref_up)
        max_contrast_pos = w_arr[np.argmax(contrast)]
        
        # PLot reflection spectrum
        plt.figure(figsize=[22, 6])
        plt.subplot(1, 3, 1)
        plt.title("Reflection spectrum")
        plt.plot(w_arr, ref_down, label="down")
        plt.plot(w_arr, ref_up, label="up")
        plt.plot([max_optval_pos, max_optval_pos], [-0.05, 1], 'r--')
        plt.plot([max_contrast_pos, max_contrast_pos], [-0.05, 1], 'r--')
        plt.ylim([0, 1])
        plt.xlabel("Frequency")
        plt.ylabel("Reflectance")
        plt.legend()

        # Plot optimization value function
        plt.subplot(1, 3, 2)
        plt.title("Optimization value function")
        plt.plot(w_arr, opt_value_fn)
        plt.plot([max_optval_pos, max_optval_pos] , [min(opt_value_fn), max(opt_value_fn)], 'r--')
        plt.xlabel("Frequency")
        plt.ylabel("Reflection Contrast")

        # Plot reflection contrast
        plt.subplot(1, 3, 3)
        plt.title("Reflection contrast spectrum")
        plt.plot(w_arr, contrast)
        plt.plot([max_contrast_pos, max_contrast_pos] , [min(contrast), max(contrast)], 'r--')
        plt.xlabel("Frequency")
        plt.ylabel("Reflection Contrast")

        if print_output:
            print(f"Maximum optimization value = {max(opt_value_fn):.3} located at frequency {max_optval_pos:.3}")
            print(f"Maximum contrast = {max(contrast):.3} located at frequency {max_contrast_pos:.3}")
            print(f"Lower reflectivity = {ref_up[np.argmax(contrast)]:.3}, higher reflectivity = {ref_down[np.argmax(contrast)]:.3}")

    def plot_reflection_contrast_empty(self, w_arr, w_up):

        # Set the detuning and computed splitting 
        c = copy.deepcopy(self)
        c.set_cavity_params({"w_c": 0})
        c.set_qubit_params({"w_up": w_up}) 
        
        ref_empty = c.reflectance(w_arr, -1)
        ref_up = c.reflectance(w_arr, 1)
        contrast = c.empty_contrast_fn(ref_empty, ref_up)
        max_contrast_pos = w_arr[np.argmax(contrast)]
        
        # PLot reflection spectrum
        plt.figure(figsize=[16, 6])
        plt.subplot(1, 2, 1)
        plt.title("Reflection spectrum")
        plt.plot(w_arr, ref_up, label="SiV")
        plt.plot(w_arr, ref_empty, label="Empty")
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
        
        print(f"Empty reflectivity = {ref_empty[np.argmax(contrast)]:.3}, SiV reflectivity = {ref_up[np.argmax(contrast)]:.3}")

class CavitySiV(Cavity):
    """ Model of a cavity with an SiV qubit placed in the cavity. The SiV is 
    modeled using the SiV class and provides the transition frequencies that the 
    cavity experiences. The SiV properties can change based on parameters such as
    the applied B field and strain. """

    def __init__(self, cavity_params=None, qubit_params=None, siv=None):

        super().__init__(cavity_params, qubit_params)

        # Remove the qubit transitions as those will be handled by the SiV object
        del self.qubit_params["w_up"]
        del self.qubit_params["w_down"]

        # Use the default SiV constructor if None is provided
        if siv is None:
            siv = SiV()
        self.siv = copy.deepcopy(siv)

        # Set transition frequency from SiV object
        self.qubit_params["w_down"] = 0
        self.qubit_params["w_up"] = self.siv.transition_splitting()

    def __repr__(self):
        return f"CavitySiV({str(self.cavity_params)}, {str(self.qubit_params)}, {str(self.siv)})"
    
    def update_siv_params(self, **siv_params):
        """ Update the SiV params with a new set of params as keyword args. """
        self.siv.update_val(**siv_params)
        self.qubit_params["w_up"] = self.siv.transition_splitting()

    def replace_siv(self, siv):
        """ Update the SiV params with a new set of params as keyword args. """
        self.siv = copy.deepcopy(siv)
        self.qubit_params["w_up"] = self.siv.transition_splitting()

    def spin_contrast(self, w, B, delta, B_axis):
        """ Function that we want to optimize over to maximize contrast.
            Will be fed into the optimization routine to find the optimal B and delta. """
        
        c = copy.deepcopy(self)
        B = B * np.array(B_axis)

        # Compute splitting at given field
        c.update_siv_params(B=B)
        splitting = c.siv.transition_splitting() 

        # Set the detuning and computed splitting
        c.set_cavity_params({"w_c": delta})
        c.set_qubit_params({"w_down": 0, "w_up": splitting}) 

        return c.spin_contrast_fn(c.reflectance(w, 0), c.reflectance(w, 1)) 

    def optimize_spin_contrast(self, w_0, B_0, delta_0, w_bounds, B_bounds, delta_bounds, B_axis):
        opt = minimize(lambda args: -self.spin_contrast(args[0], args[1], args[2], B_axis), 
                        x0=[w_0, B_0, delta_0], bounds=(w_bounds, B_bounds, delta_bounds))

        # Extract optimal params and optimal value
        w_opt, B_opt, delta_opt = opt.x
        spin_contrast_opt = -opt.fun # Negative since we used minimize()

        return ((w_opt, B_opt, delta_opt), spin_contrast_opt)

    def plot_reflection_contrast(self, w_arr, B, delta):

        # Compute splitting at given field
        c = copy.deepcopy(self)
        c.update_siv_params(B=B)
        splitting = c.siv.transition_splitting()

        # Set the detuning and computed splitting 
        c.set_cavity_params({"w_c": delta})
        c.set_qubit_params({"w_down": 0, "w_up": splitting}) 
        
        ref_down = c.reflectance(w_arr, 0)
        ref_up = c.reflectance(w_arr, 1)
        contrast = c.spin_contrast_fn(ref_down, ref_up)
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
        
        print(f"Maximum contrast = {max(contrast):.3} located at frequency {max_contrast_pos:.3}")
        print(f"Lower reflectivity = {ref_up[np.argmax(contrast)]:.3}, higher reflectivity = {ref_down[np.argmax(contrast)]:.3}")
