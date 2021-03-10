import inspect
import numpy as np
from scipy.optimize import curve_fit
from sivqed.models.cavity import Cavity, MultiQubitCavity

def single_siv_ref(w, w_up, g_up, gamma_up, 
                      w_c, k_in, k_tot, A, B):
    return A * Cavity.reflectance_fn(w, 1, 0, 0, 0, w_up, g_up, gamma_up, w_c, k_in, k_tot) + B

def two_siv_ref(w, w_up_1, g_up_1, gamma_up_1, 
                   w_up_2, g_up_2, gamma_up_2,
                   w_c, k_in, k_tot, A, B):
        return A * MultiQubitCavity.reflectance_fn(w, 1, 
                    [{"w_up": w_up_1, "g_up": g_up_1, "gamma_up": gamma_up_1},
                     {"w_up": w_up_2, "g_up": g_up_2, "gamma_up": gamma_up_2}], 
                    w_c, k_in, k_tot) + B

def three_siv_ref(w, w_up_1, g_up_1, gamma_up_1, 
                     w_up_2, g_up_2, gamma_up_2,
                     w_up_3, g_up_3, gamma_up_3,
                     w_c, k_in, k_tot, A, B):
        return A * MultiQubitCavity.reflectance_fn(w, 1, 
                    [{"w_up": w_up_1, "g_up": g_up_1, "gamma_up": gamma_up_1},
                     {"w_up": w_up_2, "g_up": g_up_2, "gamma_up": gamma_up_2},
                     {"w_up": w_up_3, "g_up": g_up_3, "gamma_up": gamma_up_3}], 
                    w_c, k_in, k_tot) + B

def fit_reflection(freqs, spectrum, fit_func=single_siv_ref, p0=None, bounds=(-np.inf, np.inf)):
    """ Fit the reflection spectrum as a function of the laser frequency sweep (freqs). 
    Returns the fitted parameters. 
    
    p0 : Array
        Initial guesses for parameters - w_up, g_up, gamma_up, w_c, k_in, k_out, k_tot
    bounds : 2-tuple
        Lower and upper bounds on parameters. Each element is either an array 
        with the length equal to the number of parameters, or a scalar 
        (in which case the bound is taken to be the same for all parameters).  
    """

    popt, cov = curve_fit(fit_func, freqs, spectrum, p0=p0, bounds=bounds)

    params = inspect.getargspec(fit_func)[0][1:] # Chop off the 1st arg ('w')
    return dict(zip(params, popt)), cov
