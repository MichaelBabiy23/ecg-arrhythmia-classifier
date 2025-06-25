# 1️⃣ Import everything we need
import wfdb                              # For reading ECG recordings & labels
from wfdb.processing import xqrs_detect, ann2rr
import numpy as np                      # For quick math with lists of numbers
import pywt                             # For wavelet transformation
import scipy.stats  # For higher-level stats (skewness, kurtosis)


# 2️⃣ Help function: convert MIT-BIH beat codes into 5 simple classes
def map_to_AAMI(symbol):
    # Normal beats
    if symbol in ['N', 'L', 'R', 'e', 'j']:
        return 'N'
    # Supraventricular ectopic beats
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'
    # Ventricular ectopic beats
    elif symbol in ['V', 'E']:
        return 'V'
    # Fusion beat
    elif symbol == 'F':
        return 'F'
    # Paced / unknown beats
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'
    else:
        return 'Q'  # Default when it's something else


# 3️⃣ Core function that reads one record and extracts features
def extract_features_for_record(rec_name):
    # A. Read ECG signal and annotation file
    full_rec_path = f"physionet.org/files/mitdb/1.0.0/{rec_name}"
    sig, fields = wfdb.rdsamp(full_rec_path, channels=[0])  # Lead I
    ann = wfdb.rdann(full_rec_path, 'atr')                  # Beat labels
    ecg = sig.flatten()
    fs = fields['fs']                                  # ~360 Hz sampling rate

    # B. Find the R-peaks (heartbeats)
    rpeaks = ann.sample

    rr_intervals = np.diff(rpeaks) / fs
    rr_intervals = np.insert(rr_intervals, 0, rr_intervals[0])  # Adds the first RR value at index 0 to fix the shape

    print(f"Length of rpeaks: {len(rpeaks)}")
    print(f"Length of annotations: {len(ann.symbol)}")
    print(f"Length of rr_intervals: {len(rr_intervals)}")

    features, labels = [], []
    window = int(0.25 * fs)  # 250 ms either side of R-peak

    # C. Loop through each beat
    for i, r in enumerate(rpeaks):
        # Extract the heartbeat segment centered around the detected R-peak
        # Ensure segment bounds are within the ECG signal limits
        seg = ecg[max(0, r - window): r + window]

        feats = []

        # 1. Simple waveform statistics for the segment
        feats.extend([len(seg), np.mean(seg), np.std(seg), np.max(seg) - np.min(seg)])

        # 2. Timing info: time between this detected beat and the next/previous
        # Uses the `rr_intervals` array derived from detected peaks.
        current_rr = rr_intervals[i]
        prev_rr = rr_intervals[i - 1] if i > 0 else rr_intervals[i]  # If first beat, prev_rr = current_rr

        feats.extend([current_rr, prev_rr])

        # 3. Wavelet decomposition + stats for each band
        coeffs = pywt.wavedec(seg, 'db1', level=3)
        for c in coeffs:
            feats.extend([np.mean(c), np.std(c)])

        # 4. Shape descriptors: skewness & kurtosis of the segment
        feats.extend([scipy.stats.skew(seg), scipy.stats.kurtosis(seg)])

        features.append(feats)
        labels.append(map_to_AAMI(ann.symbol[i]))

    return features, labels


# 4️⃣ Process **all** records in the MIT-BIH dataset
record_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]  # All 48 records from RECORDS file
all_feats, all_labels = [], []

for rec in record_ids:
    rec_str = str(rec)
    feats, labs = extract_features_for_record(rec_str)
    all_feats.extend(feats)
    all_labels.extend(labs)

# 5️⃣ Final step: Make arrays and save them for your model
X = np.array(all_feats)
y = np.array(all_labels)
np.savez('all_ecg_features.npz', X=X, y=y)

