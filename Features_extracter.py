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
    rpeaks = xqrs_detect(ecg, fs=fs)
    rr_intervals = ann2rr(
        record_name=full_rec_path, extension='atr', as_array=True
    )

    print(f"Length of rpeaks: {len(rpeaks)}")
    print(f"Length of annotations: {len(ann.symbol)}")
    print(f"Length of rr_intervals: {len(rr_intervals)}")

    features, labels = [], []
    window = int(0.25 * fs)  # 250 ms either side of R-peak

    # C. Loop through each beat
    for i in range(len(ann.symbol)):
        r = ann.sample[i]
        seg = ecg[max(0, r-window): r+window]  # The heartbeat window
        feats = []
        # 1. Simple waveform stats
        feats += [len(seg), np.mean(seg), np.std(seg), np.max(seg)-np.min(seg)]
        # 2. Timing features between beats
        # Calculate RR intervals directly from annotations
        if i > 0 and i < len(ann.sample) - 1:
            rr_current = ann.sample[i] - ann.sample[i-1]
            rr_prev = ann.sample[i-1] - (
                ann.sample[i-2] if i > 1 else ann.sample[i-1]
            )
        elif i == 0 and len(ann.sample) > 1:
            rr_current = ann.sample[i+1] - ann.sample[i]
            rr_prev = rr_current  # Placeholder for the first beat
        elif len(ann.sample) == 1:
            rr_current = 0  # No interval for a single beat
            rr_prev = 0
        else:
            rr_current = 0
            rr_prev = 0

        feats += [rr_current, rr_prev]
        # 3. Wavelet decomposition + stats for each band
        coeffs = pywt.wavedec(seg, 'db1', level=3)
        for c in coeffs:
            feats += [np.mean(c), np.std(c)]
        # 4. Shape descriptors: skewness & kurtosis
        feats += [scipy.stats.skew(seg), scipy.stats.kurtosis(seg)]
        # Save this beat's features and label
        features.append(feats)
        labels.append(map_to_AAMI(ann.symbol[i]))

    return features, labels


# 4️⃣ Process **all** records in the MIT-BIH dataset
record_ids = list(range(100, 119)) + list(range(121, 125)) + \
             list(range(200, 235))  # 48 records total
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
