import inspect
import numpy as np
from scipy.optimize import curve_fit

import lmfit

from sivqed.models.cavity import Cavity, MultiQubitCavity

def cavity_ref(w, w_c, k_in, k_tot, A, B):
    return A * Cavity.reflectance_fn(
                        w, -1, 0, 0, 0,
                        0, 0, 0,
                        w_c, k_in, k_tot
                        ) + B

def single_siv_ref(w, w_up, g_up, gamma_up, w_c, k_in, k_tot, A, B):
    return A * Cavity.reflectance_fn(
                w, 1, 0, 0, 0,
                w_up, g_up, gamma_up,
                w_c, k_in, k_tot
                ) + B

def two_siv_ref(w, w_up_1, g_up_1, gamma_up_1, w_up_2, g_up_2, gamma_up_2, w_c, k_in, k_tot, A, B):

    return A * MultiQubitCavity.reflectance_fn(w, 1,
        [{"w_up": w_up_1, "g_up": g_up_1, "gamma_up": gamma_up_1},
         {"w_up": w_up_2, "g_up": g_up_2, "gamma_up": gamma_up_2}],
        w_c, k_in, k_tot) + B

def three_siv_ref(w, w_up_1, g_up_1, gamma_up_1, w_up_2, g_up_2, gamma_up_2,
                  w_up_3, g_up_3, gamma_up_3, w_c, k_in, k_tot, A, B):

    return A * MultiQubitCavity.reflectance_fn(w, 1,
        [{"w_up": w_up_1, "g_up": g_up_1, "gamma_up": gamma_up_1},
         {"w_up": w_up_2, "g_up": g_up_2, "gamma_up": gamma_up_2},
         {"w_up": w_up_3, "g_up": g_up_3, "gamma_up": gamma_up_3}],
        w_c, k_in, k_tot) + B

cavity_ref_model = lmfit.Model(cavity_ref)
single_siv_ref_model = lmfit.Model(single_siv_ref)
two_siv_ref_model = lmfit.Model(two_siv_ref)
three_siv_ref_model = lmfit.Model(three_siv_ref)
