# import relevant packages
import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf # This will be used to create a PDF to store multiple plots in the same file

##Function to compute sample mean and covariance=========================================================================
def produce_trunc_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)
        E_val: integer, (number of electrodes)

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signal_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_all_cov, (E_val, length_per_electrode, length_per_electrode)

    note:
    -----
        descriptive mean and sample covariance statistics from real data
        In this case, E_val=16, length_per_electrode=25.
        But you should pass them as arguments or calculate them inside the function.
    """

    n_trials, n_features = input_signal.shape
    time_per_elec = n_features // E_val

    target = input_signal[input_type == 1]
    nontarget = input_signal[input_type == -1]

    target_3d = target.reshape(target.shape[0], E_val, time_per_elec)
    nontarget_3d = nontarget.reshape(nontarget.shape[0], E_val, time_per_elec)
    all_3d = input_signal.reshape(n_trials, E_val, time_per_elec)

    # means (across trials)
    tar_mean = np.mean(target_3d, axis=0)
    ntar_mean = np.mean(nontarget_3d, axis=0)

    # covariances per electrode
    tar_cov = np.array([np.cov(target_3d[:, e, :], rowvar=False) for e in range(E_val)])
    ntar_cov = np.array([np.cov(nontarget_3d[:, e, :], rowvar=False) for e in range(E_val)])
    all_cov = np.array([np.cov(all_3d[:, e, :], rowvar=False) for e in range(E_val)])

    return [tar_mean, ntar_mean, tar_cov, ntar_cov, all_cov]

##Function to plot sample mean=========================================================================
def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    r"""
    :param eeg_tar_mean:
    :param eeg_ntar_mean:
    :param subject_name:
    :param time_index:
    :param E_val:
    :param electrode_name_ls:
    :param y_limit: optional parameter, a list or an array of two numbers
    :param fig_size: optional parameter, a tuple of two numbers
    :return:
    """

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.ravel()

    order = list(range(E_val))

    for pos in range(E_val):
        ch = order[pos]
        ax = axes[pos]
        ax.plot(time_index, eeg_tar_mean[ch], color='red', label='Target')  # red = target
        ax.plot(time_index, eeg_ntar_mean[ch], color='blue', label='Non-Target')  # blue = non-target
        ax.set_title(electrode_name_ls[ch])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_ylim(y_limit)
        ax.grid(alpha=0.3, linestyle=':')

    for j in range(E_val, 16):
        axes[j].axis('off')

    axes[0].legend(loc='upper right')
    fig.suptitle(f"{subject_name} Target v. Non-Target Sample Means", fontsize=14, y=0.98)
    plt.tight_layout()


##Function to plot sample covariance=========================================================================
# function makes a 4x4 grid heatmap showing covariance matrix/how timpts move togethr
# e-val = # electrodes

def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14, 12)  #
):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.ravel()  # flattens 2d grid (4x4) into 1d list of 16 plots

    order = list(range(E_val))  # fills subplots in the oG electrode order

    T = len(time_index)  # = how many time pts per electrode (len = counts how many items)
    X, Y = np.meshgrid(time_index,
                       time_index)  # makes 2 2d grids that tell the plot which x,y coord go together (both will be 25*25 array representing combos of timepts)

    vals = np.concatenate([eeg_cov[e].ravel() for e in
                           range(E_val)])  # long 1d list of numbers -> giant array of 16 lists = shows covar vals
    vmin, vmax = np.percentile(vals, [2, 98])  # lower and upper color limits of heatmpa

    # draws each electrode's heatmap
    for pos in range(E_val):
        ch = order[pos]  # gets the index that should go into that position
        ax = axes[pos]  # chooses the subplot (axes object) for that position)
        im = ax.contourf(X, Y, eeg_cov[ch], levels=20, cmap='coolwarm', vmin=vmin,
                         vmax=vmax)  # makes a filled contour map
        ax.set_title(electrode_name_ls[ch])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Time (ms)")
        ax.invert_yaxis()  # makes y axis increase from top -> bottom

    for j in range(E_val, 16):
        axes[j].axis('off')  # turn off any unused plots

    cbar = fig.colorbar(im, ax=axes[:E_val], shrink=0.85)  # colorbar, 85% to make fit better
    cbar.set_label("Covariance (μV²)")

    fig.suptitle(f"{subject_name} — {cov_type} Sample Covariance", fontsize=14, y=0.98)
    plt.tight_layout()
