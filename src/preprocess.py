# src/preprocess.py
import numpy as np
from scipy.signal import welch
import mne

def bandpass_filter(raw, l_freq=1.0, h_freq=40.0):
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

def compute_psd(data, sfreq=250, fmin=8, fmax=30, nperseg=256):
    # data shape: (n_channels, n_times)
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    # select bands
    idx = (freqs >= fmin) & (freqs <= fmax)
    return freqs[idx], psd[..., idx]

# simple normalization
def normalize_epoch(epoch):
    # epoch: (channels, times)
    mu = epoch.mean(axis=1, keepdims=True)
    sigma = epoch.std(axis=1, keepdims=True) + 1e-6
    return (epoch - mu) / sigma

# optional: ICA pipeline for artifact removal (EOG preservation note)

def run_ica(raw, n_components=20, reject=None):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    return ica
