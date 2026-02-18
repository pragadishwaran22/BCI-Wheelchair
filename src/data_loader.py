import mne
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from scipy import signal


class BCI2aDataset(Dataset):
    """Simple Dataset wrapper for precomputed epochs and labels.
    X shape: (n_epochs, n_channels, n_times)
    y shape: (n_epochs,)
    """
    def __init__(self, data_array, labels, transform=None):
        self.X = data_array.astype('float32')
        self.y = labels.astype('int64')
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def _preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freqs=(50.0,), apply_ica=True,
                    ica_n_components=0.99, reject_tmax_uv=200.0):
    """Apply notch, bandpass, avg ref, ICA (if possible). Returns cleaned Raw."""
    raw.load_data()
    # Notch: handle single value or iterable
    if notch_freqs is not None:
        try:
            raw.notch_filter(freqs=notch_freqs, picks='eeg', verbose=False)
        except Exception:
            # some mne versions accept different arg types; ignore failures
            pass

    # Bandpass
    try:
        raw.filter(l_freq, h_freq, picks='eeg', method='iir', verbose=False)
    except Exception:
        # fallback to default filter behaviour if iir fails
        raw.filter(l_freq, h_freq, picks='eeg', verbose=False)

    # Re-reference to average EEG
    try:
        raw.set_eeg_reference('average', projection=False, verbose=False)
    except Exception:
        pass

    # ICA to remove EOG artifacts if EOG channels exist or to attempt blind removal
    if apply_ica:
        picks_eeg = mne.pick_types(raw.info, eeg=True, eog=True)
        if len(picks_eeg) > 0:
            try:
                ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=42, max_iter='auto')
                ica.fit(raw, picks=picks_eeg, verbose=False)
                # try to find EOG components automatically (if EOG channels are present)
                eog_inds = []
                eog_chs = mne.pick_types(raw.info, eog=True)
                if len(eog_chs) > 0:
                    try:
                        eog_inds, scores = ica.find_bads_eog(raw, verbose=False)
                        ica.exclude = list(set(ica.exclude).union(set(eog_inds)))
                    except Exception:
                        pass
                # apply ICA (skip if no components)
                if len(ica.exclude) > 0:
                    try:
                        ica.apply(raw, verbose=False)
                    except Exception:
                        pass
            except Exception:
                # if ICA fails, continue without ICA
                pass
    return raw


def load_gdf_epochs(gdf_path, picks_eeg=None, tmin=2.0, tmax=6.0, event_id=None, sfreq=250,
                    l_freq=1.0, h_freq=40.0, notch_freqs=(50.0,), apply_ica=True,
                    reject_tmax_uv=200.0):
    """
    Load GDF and apply robust preprocessing and artifact removal.

    Returns:
      data: (n_epochs, n_channels, n_times) in Volts
      labels: (n_epochs,)
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    if picks_eeg is None:
        picks_eeg = mne.pick_types(raw.info, eeg=True, eog=True)
    # preprocess raw (notch, bandpass, avg-ref, ICA)
    raw = _preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq, notch_freqs=notch_freqs,
                          apply_ica=apply_ica, reject_tmax_uv=reject_tmax_uv)
    # extract events and epochs
    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        picks=picks_eeg, baseline=None, preload=True, verbose=False)
    # reject large artifacts (peak-to-peak)
    try:
        epochs.drop_bad(reject=dict(eeg=reject_tmax_uv * 1e-6), verbose=False)
    except Exception:
        # fallback: manual ptp-based drop
        data_ = epochs.get_data()
        ptps = data_.ptp(axis=2)
        good = (ptps < (reject_tmax_uv * 1e-6)).all(axis=1)
        epochs = epochs[good]
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    return data, labels


def load_csv_file(csv_path, seq_len=1000, overlap=700, sfreq=250,
                  l_freq=1.0, h_freq=40.0, notch_freqs=(50.0,), apply_ica=True,
                  reject_tmax_uv=200.0):
    """
    Load CSV-exported BCI2a-like files and apply preprocessing.

    Expected CSV columns: patient, time, label, epoch, EEG-*
    Behavior:
      - Groups samples by `epoch` column
      - Converts whole CSV to an MNE RawArray, runs the same preprocessing as GDF
      - For each epoch, extracts sliding windows of length `seq_len` with `overlap`
      - Applies peak-to-peak rejection on each window (threshold in microvolts)

    Notes:
      - If label text values differ, extend `label_map` accordingly.
    """
    print(f"Loading CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map expected text labels to integers for BCI2a
    label_map = {
        'left': 0,
        'right': 1,
        'foot': 2,
        'tongue': 3,
    }
    if 'label' not in df.columns:
        raise ValueError("CSV missing 'label' column")
    df['label_num'] = df['label'].map(label_map)
    # If any labels were not mapped (NaN), drop those rows
    if df['label_num'].isnull().any():
        n_bad = df['label_num'].isnull().sum()
        print(f"Warning: {n_bad} rows have unknown label values and will be skipped")
        df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)

    # Find EEG columns
    eeg_columns = [c for c in df.columns if c.startswith('EEG-')]
    if len(eeg_columns) == 0:
        raise ValueError('No EEG-* columns found in CSV file')

    # Sort rows to preserve temporal order
    df = df.sort_values(['epoch', 'time']).reset_index(drop=True)

    # Create raw from full CSV EEG data so preprocessing (ICA/filter) can be applied globally
    eeg_data = df[eeg_columns].to_numpy().T  # shape (n_channels, n_samples)
    # Detect units: if values appear large (>1e3) assume microvolts and convert to Volts for MNE
    if np.max(np.abs(eeg_data)) > 1e3:
        eeg_data = eeg_data * 1e-6  # µV -> V

    ch_types = ['eeg'] * eeg_data.shape[0]
    info = mne.create_info(ch_names=eeg_columns, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # Run the same preprocessing pipeline as for GDF
    raw = _preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq, notch_freqs=notch_freqs,
                          apply_ica=apply_ica, reject_tmax_uv=reject_tmax_uv)

    # Extract cleaned data back to numpy for epoch/window extraction
    cleaned = raw.get_data()  # (n_channels, n_samples)

    X_list = []
    y_list = []

    # For each epoch id, locate corresponding row indices in df and extract windows
    for epoch in df['epoch'].unique():
        epoch_idx = df.index[df['epoch'] == epoch].to_numpy()
        if epoch_idx.size == 0:
            continue
        start_idx = epoch_idx[0]
        end_idx = epoch_idx[-1] + 1
        eeg_epoch = cleaned[:, start_idx:end_idx]  # (n_channels, n_samples)
        n_samples = eeg_epoch.shape[1]
        if n_samples <= 0:
            continue

        # window length for this epoch: cannot exceed available samples
        win_len = min(seq_len, n_samples)
        eff_overlap = min(overlap, max(0, win_len - 1))
        stride = max(1, win_len - eff_overlap)

        label_num = int(df.loc[start_idx, 'label_num'])

        # sliding windows within epoch
        for start in range(0, n_samples - win_len + 1, stride):
            window = eeg_epoch[:, start:start + win_len]
            # reject windows with excessive peak-to-peak amplitude (threshold in microvolts)
            ptp_uv = (window.ptp(axis=1) * 1e6)  # convert V -> µV for thresholding
            if np.any(ptp_uv > reject_tmax_uv):
                continue
            X_list.append(window.astype('float32'))
            y_list.append(label_num)

    if len(X_list) == 0:
        print('No valid epochs/windows found with required length after preprocessing/rejection')
        return np.array([], dtype='float32'), np.array([], dtype='int32')

    X = np.stack(X_list, axis=0)  # (n_windows, n_channels, seq_len)
    y = np.array(y_list, dtype='int32')

    print(f'Created {len(X)} windows with shape {X.shape}')
    try:
        print('Label distribution:', np.bincount(y))
    except Exception:
        pass

    return X, y
import mne
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from scipy import signal


class BCI2aDataset(Dataset):
    """Simple Dataset wrapper for precomputed epochs and labels.
    X shape: (n_epochs, n_channels, n_times)
    y shape: (n_epochs,)
    """
    def __init__(self, data_array, labels, transform=None):
        self.X = data_array.astype('float32')
        self.y = labels.astype('int64')
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_gdf_epochs(gdf_path, picks_eeg=None, tmin=2.0, tmax=6.0, event_id=None, sfreq=250):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    if picks_eeg is None:
        picks_eeg = mne.pick_types(raw.info, eeg=True, eog=True)
    raw.filter(1., 40., picks=picks_eeg, method='iir')
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        picks=picks_eeg, baseline=None, preload=True)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    return data, labels


def load_csv_file(csv_path, seq_len=1000, overlap=900):
    """
    Load CSV-exported BCI2a-like files.

    Expected CSV columns: patient, time, label, epoch, EEG-*
    Behavior:
      - Groups samples by `epoch` column
      - For each epoch, extracts sliding windows of length `seq_len` with `overlap`
      - Returns X: (n_windows, n_channels, seq_len), y: (n_windows,)

    Notes:
      - If label text values differ, extend `label_map` accordingly.
    """
    print(f"Loading CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map expected text labels to integers for BCI2a
    # The CSVs in this dataset use: 'left', 'right', 'tongue', 'foot'
    label_map = {
        'left': 0,
        'right': 1,
        'foot': 2,
        'tongue': 3,
    }
    df['label_num'] = df['label'].map(label_map)
    # If any labels were not mapped (NaN), drop those rows
    if df['label_num'].isnull().any():
        n_bad = df['label_num'].isnull().sum()
        print(f"Warning: {n_bad} rows have unknown label values and will be skipped")
        df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)

    # Find EEG columns
    eeg_columns = [c for c in df.columns if c.startswith('EEG-')]
    if len(eeg_columns) == 0:
        raise ValueError('No EEG-* columns found in CSV file')

    # Ensure proper ordering
    df = df.sort_values(['epoch', 'time'])

    X_list = []
    y_list = []

    # For each epoch, adapt window length to available samples so we always extract at least one window
    for epoch in df['epoch'].unique():
        epoch_df = df[df['epoch'] == epoch]
        eeg_data = epoch_df[eeg_columns].to_numpy().T  # (n_channels, n_samples)
        n_samples = eeg_data.shape[1]
        if n_samples <= 0:
            continue

        # window length for this epoch: cannot exceed available samples
        win_len = min(seq_len, n_samples)
        # effective overlap cannot be >= win_len
        eff_overlap = min(overlap, max(0, win_len - 1))
        stride = max(1, win_len - eff_overlap)

        label_num = epoch_df['label_num'].iloc[0]

        # sliding windows within epoch (at least one window will be produced)
        for start in range(0, n_samples - win_len + 1, stride):
            window = eeg_data[:, start:start + win_len]
            # if win_len != seq_len we may want to pad or keep varying lengths; here we keep varying lengths
            X_list.append(window.astype('float32'))
            y_list.append(int(label_num))

    if len(X_list) == 0:
        print('No valid epochs/windows found with required length')
        return np.array([], dtype='float32'), np.array([], dtype='int32')

    X = np.stack(X_list, axis=0)  # (n_windows, n_channels, seq_len)
    y = np.array(y_list, dtype='int32')

    print(f'Created {len(X)} windows with shape {X.shape}')
    # print label distribution
    try:
        print('Label distribution:', np.bincount(y))
    except Exception:
        pass

    return X, y