import inspect
import numpy as np
from scipy.optimize import curve_fit

import lmfit

from sivqed.models.cavity import Cavity, MultiQubitCavity

# w, w_c, k_in, k_tot, A, B
def cavity_ref(params, w, spectrum):
    params = params.valuesdict()

    return params["A"] * Cavity.reflectance_fn(
                        w, -1, 0, 0, 0, 
                        0, 0, 0, 
                        params["w_c"], params["k_in"], params["k_tot"]
                        ) + params["B"] - spectrum

# w, w_up, g_up, gamma_up, w_c, k_in, k_tot, A, B 
def single_siv_ref(params, w, spectrum):
    params = params.valuesdict()

    return params["A"] * Cavity.reflectance_fn(
                w, 1, 0, 0, 0, 
                params["w_up"], params["g_up"], params["gamma_up"], 
                params["w_c"], params["k_in"], params["k_tot"]
                ) + params["B"] - spectrum

# w, w_up_1, g_up_1, gamma_up_1, w_up_2, g_up_2, gamma_up_2, w_c, k_in, k_tot, A, B
def two_siv_ref(params, w, spectrum):
    params = params.valuesdict()

    return params["A"] * MultiQubitCavity.reflectance_fn(w, 1, 
        [{"w_up": params["w_up_1"], "g_up": params["g_up_1"], "gamma_up": params["gamma_up_1"]},
         {"w_up": params["w_up_2"], "g_up": params["g_up_2"], "gamma_up": params["gamma_up_2"]}], 
        params["w_c"], params["k_in"], params["k_tot"]) + params["B"] - spectrum

# w, w_up_1, g_up_1, gamma_up_1, w_up_2, g_up_2, gamma_up_2,
# w_up_3, g_up_3, gamma_up_3, w_c, k_in, k_tot, A, B
def three_siv_ref(params, w, spectrum):
    params = params.valuesdict()

    return params["A"] * MultiQubitCavity.reflectance_fn(w, 1, 
        [{"w_up": params["w_up_1"], "g_up": params["g_up_1"], "gamma_up": params["gamma_up_1"]},
         {"w_up": params["w_up_2"], "g_up": params["g_up_2"], "gamma_up": params["gamma_up_2"]},
         {"w_up": params["w_up_3"], "g_up": params["g_up_3"], "gamma_up": params["gamma_up_3"]}], 
        params["w_c"], params["k_in"], params["k_tot"]) + params["B"] - spectrum

def fit_reflection(params, freqs, spectrum, fit_func, method='leastsq'):
    """ Fit the reflection spectrum as a function of the laser frequency sweep (freqs). 
    Returns the fitted parameters. 

    params : Parameters object
        Stores the list of parameters and associated bounds
    freqs : Numpy array
        List of frequencies (x-axis) to be fitted
    spectrum : Numpy array
        List of counts (y-axis) to be fitted
    fit_func : Function
        Function to be used for fitting
    """

    result = lmfit.minimize(fit_func, params, args=(freqs, spectrum), method=method)
    return result


