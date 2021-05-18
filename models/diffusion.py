import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def diffused_siv_peaks(cavity, detuning_freqs, diffusion_stdev, plot=True):
    siv_min = []
    siv_max = []
    diffused_siv_min = []
    diffused_siv_max = []
    
    # Number of steps in the freqs_filtered mesh so that it has
    # enough frequency resolution for the specified diffusion Gaussian
    # 30 comes from freqs_filtered width of 30
    # 5 comes from our desire for the Gaussian to have at least 5 stdev width in index space
    num_freq_steps = max(int(30 / (diffusion_stdev / 5)),  2000)
    freq_step = 30 / num_freq_steps
    
    if num_freq_steps > 10000:
        print("WARNING: The diffusion stdev is very small, so we will need a very fine frequency resolution to compute the convolution.")
        print(f"The current value of diffusion of {diffusion_stdev} requires {num_freq_steps} freq steps.")
    
    # Normalization to convert from stdev in freq-space to
    # stdev in the index value (depends on the freq-list step)
    diffusion_stdev_idx = diffusion_stdev / freq_step

    for idx, detuning in enumerate(detuning_freqs):
        # Set a particular value of detuning
        cavity.set_qubit_params({"w_up" : detuning})
        
        # Only crop the spectrum around the SiV resonance
        # This is necessary when computing the min/max for contrast, 
        # otherwise it would take the global min/max
        freqs_filtered = np.linspace(detuning - 15, detuning + 15, num_freq_steps)
        siv_filtered = cavity.reflectance(freqs_filtered, 1)
        
        # Compute the diffused spectrum
        diffused_siv_filtered = gaussian_filter1d(siv_filtered, diffusion_stdev_idx)
        
        # Plot the undiffused spectra and the diffused spectra at alternate spots
        if plot:
            if idx % 100 == 0:
                plt.plot(freqs_filtered, siv_filtered, 'b--', lw=1)
            if idx % 100 == 50:
                plt.plot(freqs_filtered, diffused_siv_filtered, 'g', lw=1)

        siv_min.append(min(siv_filtered))
        siv_max.append(max(siv_filtered))
        diffused_siv_min.append(min(diffused_siv_filtered))
        diffused_siv_max.append(max(diffused_siv_filtered))

    siv_min, siv_max = np.array(siv_min), np.array(siv_max)
    diffused_siv_min, diffused_siv_max = np.array(diffused_siv_min), np.array(diffused_siv_max)
    
    return siv_min, siv_max, diffused_siv_min, diffused_siv_max

def plot_diffused_contrasts(detuning_freqs, diffusion_stdev_list, diffused_siv_min_list, diffused_siv_max_list, cavity_min):
    
    diffused_siv_min_list = np.array(diffused_siv_min_list)
    diffused_siv_max_list = np.array(diffused_siv_max_list)
    
    # Compute contrasts
    diffused_siv_contrast_list = diffused_siv_max_list / diffused_siv_min_list
    
    fig, ax = plt.subplots(1, 2, figsize=[16, 5])
    for idx, diffusion_stdev in enumerate(diffusion_stdev_list):
        diffused_siv_min = diffused_siv_min_list[idx]
        diffused_siv_contrast = diffused_siv_contrast_list[idx]

        ax[0].plot(detuning_freqs, diffused_siv_min / cavity_min)
        ax[1].plot(detuning_freqs, diffused_siv_contrast, label=f"Diffusion = {diffusion_stdev:.1e}")

    ax[0].axhline(0.0, c='k', ls='--', lw=1.5)
    ax[0].axhline(1.0, c='k', ls='--', lw=1.5)
    ax[0].set_xlim(0, 250)
    ax[0].set_xlabel("SiV detuning")
    ax[0].set_ylabel("SiV dip ratio to cavity min")

    ax[1].set_xlabel("SiV detuning")
    ax[1].set_ylabel("SiV peak contrast")
    ax[1].set_xlim(0, 250)
    ax[1].set_ylim(0, 20)

    fig.legend(bbox_to_anchor=(1.2, 0.9))
    fig.tight_layout()
    
    return fig, ax