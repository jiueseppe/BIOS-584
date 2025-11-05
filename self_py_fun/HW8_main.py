# import relevant packages
import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf # This will be used to create a PDF to store multiple plots in the same file

#import funtions from HW8FUN.py
from HW8Fun import produce_trunc_mean_cov, plot_trunc_mean, plot_trunc_cov

#Global Constants and parent Directory=============================================================
bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

parent_dir = 'C:/Users/josep/Documents/GitHub/BIOS_584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25) # This is a hypothetic time range up to 800 ms after each stimulus.

subject_name = 'K114'
session_name = '001_BCI_TRN'

#New Directory (if doens't exist)==========================================================================
new_dir = os.path.join(parent_dir, subject_name)

if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    print("new folder: {new_dir}")
else:
    print("folder exists already: {new_dir}")

#Load Matlab file============================================================================================
matlab_file_path = os.path.join(parent_data_dir, 'K114_001_BCI_TRN_Truncated_Data_0.5_6.mat')

eeg_trunc_obj = sio.loadmat(matlab_file_path)

print(eeg_trunc_obj.keys())

eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = eeg_trunc_obj['Type']

eeg_trunc_type = np.squeeze(eeg_trunc_type, axis=None)

vals = np.asarray(eeg_trunc_type).ravel()
uniq, counts = np.unique(vals, return_counts=True)
print("Unique codes  counts:")
for u, c in zip(uniq, counts):
    print(f"  {u}: {c}")


print(eeg_trunc_signal.shape)
print(eeg_trunc_type.shape)

print(eeg_trunc_type[:10])
print("eeg_trunc_type contains EEG trial labels; 1 indicates target stimuli and –1 indicates non-target stimuli.")

#compute mean and covariances======================================================
tar_mean, ntar_mean, tar_cov, ntar_cov, all_cov = produce_trunc_mean_cov(
    eeg_trunc_signal, eeg_trunc_type, electrode_num
)

#Mean (target vs. non-target) - call plot_trunc_mean
plot_trunc_mean(
    tar_mean, ntar_mean,
    subject_name, time_index,
    electrode_num, electrode_name_ls)
plt.savefig(os.path.join(new_dir, "Mean.png"), dpi=200, bbox_inches="tight")
plt.close()

#Covariance (Target) - call plot_trunc_cov for target covariance only
plot_trunc_cov(
    tar_cov, "Target",
    time_index, subject_name,
    electrode_num, electrode_name_ls)
plt.savefig(os.path.join(new_dir, "Covariance_Target.png"), dpi=200, bbox_inches="tight")
plt.close()

#Covariance (non-target) - call plot_trunc_cov for non-target covariance only
plot_trunc_cov(
    ntar_cov, "Non-target",
    time_index, subject_name,
    electrode_num, electrode_name_ls)
plt.savefig(os.path.join(new_dir, "Covariance_Non-Target.png"), dpi=200, bbox_inches="tight")
plt.close()

#Covariance (ALL targets) - call plot_trunc_cov for all covariance only
plot_trunc_cov(
    all_cov, "All",
    time_index, subject_name,
    electrode_num, electrode_name_ls)
plt.savefig(os.path.join(new_dir, "Covariance_All.png"), dpi=200, bbox_inches="tight")
plt.close()