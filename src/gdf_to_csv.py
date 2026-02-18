import os
import mne
import numpy as np
import pandas as pd


def gdf_epoch_to_csv(gdf_path, out_csv_path, tmin=2.0, tmax=6.0, picks_eeg=None, event_code_map=None):
    """Convert a single GDF file into a CSV where rows are samples and columns are EEG channels.

    CSV columns: patient, time, label, epoch, EEG-<chname>...
    - time: sample index within epoch (0-based)
    - epoch: integer epoch id within the file (0-based)
    - label: one of 'left','right','foot','tongue'

    event_code_map: dict mapping raw event codes (integers) to label strings.
    """
    print(f"Converting {gdf_path} -> {out_csv_path}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    if picks_eeg is None:
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
    else:
        picks = picks_eeg

    # get channel names for EEG picks
    ch_names = [raw.ch_names[i] for i in picks]

    # Build events array from annotations but only keep task-related codes.
    # BCIC IV-2a task event codes are commonly: 769=left, 770=right, 771=foot, 772=tongue
    sfreq = raw.info.get('sfreq', 250.0)
    ann = raw.annotations
    # prefer canonical BCI IV-2a task codes 769-772
    default_target_codes = {'769', '770', '771', '772'}
    events_list = []

    # Try building events directly from annotations; prefer canonical task codes.
    for on, dur, desc in zip(ann.onset, ann.duration, ann.description):
        desc_str = str(desc)
        if desc_str in default_target_codes:
            sample = int(round(on * sfreq))
            events_list.append([sample, 0, int(desc_str)])

    # If we found direct task codes, deduplicate by sample (in case multiple ann at same sample)
    if len(events_list) > 0:
        # group by sample and pick one code per sample (prefer any 769-772)
        by_sample = {}
        for s, z, code in events_list:
            by_sample.setdefault(s, []).append(int(code))
        dedup = []
        for s, codes in sorted(by_sample.items()):
            # prefer codes in canonical set (already ensured), otherwise pick first
            chosen = codes[0]
            dedup.append([s, 0, int(chosen)])
        events = np.array(dedup, dtype=int)
    else:
        # fallback: use mne.events_from_annotations but allow merging repeated events
        try:
            events, annot_event_map = mne.events_from_annotations(raw, event_repeated='merge')
        except TypeError:
            # older mne versions may not have event_repeated kwarg; try default call
            events, annot_event_map = mne.events_from_annotations(raw)
        # filter events to keep only those matching canonical or provided event_code_map keys
        allowed_codes = set()
        if event_code_map is not None:
            allowed_codes = set(int(k) for k in event_code_map.keys())
        else:
            allowed_codes = set([769, 770, 771, 772])
        # keep only events whose code is in allowed_codes
        if events.size == 0:
            events = np.empty((0, 3), dtype=int)
        else:
            mask = np.isin(events[:, 2], list(allowed_codes))
            events = events[mask]

    # if still no events, print available annotation descriptions and skip file
    if events.shape[0] == 0:
        unique_desc = sorted(set(str(d) for d in ann.description))
        print(f"Used Annotations descriptions: {unique_desc}")
        print(f"No matching task events found in {gdf_path}; skipping.")
        return

    # default mapping if none provided (BCI IV-2a canonical mapping)
    if event_code_map is None:
        event_code_map = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue'}

    # create epochs using filtered events
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=None, preload=True, verbose=False)

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    evts = epochs.events[:, -1]

    rows = []
    for ep_idx in range(X.shape[0]):
        label_code = int(evts[ep_idx])
        label = event_code_map.get(label_code, str(label_code))
        epoch_data = X[ep_idx]  # shape (n_ch, n_t)
        n_t = epoch_data.shape[1]
        # for each time sample, create row with EEG-<ch> columns
        for t in range(n_t):
            row = {
                'patient': os.path.basename(gdf_path),
                'time': int(t),
                'label': label,
                'epoch': int(ep_idx)
            }
            for ch_i, ch_name in enumerate(ch_names):
                row[f'EEG-{ch_name}'] = float(epoch_data[ch_i, t])
            rows.append(row)

    if len(rows) == 0:
        print(f"No epochs extracted from {gdf_path}")
        return

    df = pd.DataFrame(rows)
    # ensure consistent column order: patient,time,label,epoch,EEG-*
    eeg_cols = [c for c in df.columns if c.startswith('EEG-')]
    df = df[['patient', 'time', 'label', 'epoch'] + eeg_cols]
    df.to_csv(out_csv_path, index=False)
    print(f"Wrote {len(df)} rows to {out_csv_path}")


def convert_all_in_dir(data_dir='data', out_dir='data', tmin=2.0, tmax=6.0):
    os.makedirs(out_dir, exist_ok=True)
    gdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.gdf')]
    if not gdf_files:
        print('No .gdf files found in', data_dir)
        return
    for gf in sorted(gdf_files):
        in_path = os.path.join(data_dir, gf)
        out_name = os.path.splitext(gf)[0] + '.csv'
        out_path = os.path.join(out_dir, out_name)
        try:
            gdf_epoch_to_csv(in_path, out_path, tmin=tmin, tmax=tmax)
        except Exception as e:
            print(f"Failed to convert {in_path}: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory with .gdf files')
    parser.add_argument('--out-dir', default='data', help='Directory to write .csv files')
    parser.add_argument('--tmin', type=float, default=2.0)
    parser.add_argument('--tmax', type=float, default=6.0)
    args = parser.parse_args()
    convert_all_in_dir(args.data_dir, args.out_dir, tmin=args.tmin, tmax=args.tmax)
