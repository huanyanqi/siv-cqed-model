import numpy as np
from scipy.optimize import curve_fit

import corner
import lmfit
import matplotlib.pyplot as plt

from sivqed.models.cavity import Cavity, MultiQubitCavity

def cavity_ref(w, w_c, k_in, k_tot, A, B):

    return A * Cavity.reflectance_fn(w, -1,
        [dict()], w_c, k_in, k_tot) + B


def single_siv_ref(w, w_up, g_up, gamma_up, w_c, k_in, k_tot, A, B):

    return A * Cavity.reflectance_fn(w, 1,
        [{"w_up": w_up, "g_up": g_up, "gamma_up": gamma_up}],
        w_c, k_in, k_tot) + B

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

def mcmc(freqs, counts, model, results, data_noise, input_params=None, plot=True):
    # Add additional (nuisance) parameter to model the data uncertainty
    # This range should correspond to the expected noise in the data
    noise_exp, noise_min, noise_max = data_noise
    results.params.add('__lnsigma', value=np.log(noise_exp), min=np.log(noise_min), max=np.log(noise_max))
    
    # Default emcee parameters, can be overwritten by input_params
    emcee_params = {"is_weighted": False, "burn": 300, "steps": 6000, "thin": 1}
    if input_params is not None: 
        emcee_params.update(input_params)
    emcee_results = model.fit(counts, results.params, w=freqs, method="emcee", fit_kws=emcee_params)
    
    # Check the autocorrelation time, guideline is num_steps should be 50 x t_cor
    try:
        print(f"Autocorrelation times: {emcee_results.acor}")
        print(f"Max 50 x Autocorrelation times: {max(50 * emcee_results.acor)}\n")
    except AttributeError:
        pass

    fitted_var_names = [param.name for param in emcee_results.params.values() if param.vary]
    fitted_values = [param.value for param in emcee_results.params.values() if param.vary]
    
    if plot:
        # Plot the corner plot for how the variables relate to one another
        emcee_plot = corner.corner(emcee_results.flatchain, labels=emcee_results.var_names, truths=fitted_values)

        # General rule of thumb is that acceptance fraction should be ~0.25
        plt.figure()
        plt.plot(emcee_results.acceptance_fraction, 'b') 

    # Print the median point from the MCMC chains
    print('Median of posterior probability distribution')
    print('--------------------------------------------')
    lmfit.report_fit(emcee_results.params, min_correl=0.4)
    
    # Obtain the Maximum Likelihood Estimation
    highest_prob_idx = np.argmax(emcee_results.lnprob)
    mle_soln = emcee_results.chain[np.unravel_index(highest_prob_idx, emcee_results.lnprob.shape)]
    # Copy the original results object and replace with MLE values
    emcee_mle_params = emcee_results.params.copy()
    for i, name in enumerate(fitted_var_names):
        emcee_mle_params[name].value = mle_soln[i]

    print('\nMaximum Likelihood Estimation from emcee       ')
    print('-------------------------------------------------')
    print('Parameter  MLE Value   Median Value   Uncertainty')
    for name in fitted_var_names:
        print(f'  {name:5s}  {emcee_mle_params[name].value:11.5f} {emcee_results.params[name].value:11.5f}   {emcee_results.params[name].stderr:11.5f}')
    
    # Return median and max likelihood results
    return emcee_results, emcee_mle_params

cavity_ref_model = lmfit.Model(cavity_ref)
single_siv_ref_model = lmfit.Model(single_siv_ref)
two_siv_ref_model = lmfit.Model(two_siv_ref)
three_siv_ref_model = lmfit.Model(three_siv_ref)
