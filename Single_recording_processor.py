# 1️⃣ Load helpful tools
import wfdb  # ECG file reading
from wfdb.processing import xqrs_detect, ann2rr  # Heartbeat detection & RR intervals
import numpy as np  # Numerical ops
import pywt  # Wavelet transforms
import scipy.stats  # Statistical functions


# 2️⃣ Turn special beat codes into easy names
def map_to_AAMI(symbol):
    if symbol in ['N', 'L', 'R', 'e', 'j']:
        return 'N'  # Normal
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'  # Supraventricular
    elif symbol in ['V', 'E']:
        return 'V'  # Ventricular
    elif symbol == 'F':
        return 'F'  # Fusion (mixed beat)
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'  # Unknown or paced
    else:
        return 'Q'


# 3️⃣ Process just record "100"
record = wfdb.rdsamp('physionet.org/files/mitdb/1.0.0/100', channels=[0])  # Read ECG signal
ann = wfdb.rdann('physionet.org/files/mitdb/1.0.0/100', 'atr')  # Read beat labels
ecg = record[0].flatten()  # Flatten ECG signal
fs = record[1]['fs']  # Sampling frequency

# 4️⃣ Find every heartbeat spike (R peak)
rpeaks = xqrs_detect(ecg, fs=fs)  # Pan–Tompkins R-peak detection
# This method uses filters and special math to pick out real heartbeats

# 5️⃣ Measure time gaps between beats
rr_intervals = ann2rr(record_name='physionet.org/files/mitdb/1.0.0/100', extension='atr', as_array=True)

# 6️⃣ Prepare lists to save data for each beat
beat_features = []
beat_labels = []

# 7️⃣ For each heartbeat, grab info
window = int(0.25 * fs)  # Save 0.25 sec before/after spike

for i, r in enumerate(rpeaks):
    seg = ecg[max(0, r-window): r+window]  # The heartbeat window
    feats = []
    # ➕ Basic stats: how long, average height, wiggle amount, top-to-bottom height
    feats.extend([
        len(seg),  # Number of points
        np.mean(seg),  # Average height
        np.std(seg),  # Wiggle amount
        np.max(seg) - np.min(seg)  # Height range
    ])

    # ⏱ Timing info: time between this beat and the next/previous
    feats.append(rr_intervals[i])
    feats.append(rr_intervals[i-1] if i > 0 else rr_intervals[i])

    # 🎵 Wavelet stats: split into layers and get average & wiggle level
    coeffs = pywt.wavedec(seg, 'db1', level=3)
    for c in coeffs:
        feats.extend([np.mean(c), np.std(c)])

    # 🔢 Measure skewness (tilt) and kurtosis (peakiness)
    feats.append(scipy.stats.skew(seg))
    feats.append(scipy.stats.kurtosis(seg))

    beat_features.append(feats)
    beat_labels.append(map_to_AAMI(ann.symbol[i]))

# 8️⃣ Turn lists into arrays for model training
X = np.array(beat_features)
y = np.array(beat_labels)

# 🧾 Now we have:
# X = features (numbers) for each beat
# y = labels (class letter) for each beat

print("Beats processed:", len(X))
print("Feature vector length:", X.shape[1])
print(X.shape)
print("Unique beat labels:", set(y))

# Calculate and print average heart rate and recording length
avg_rr_interval_seconds = np.mean(rr_intervals) / fs
avg_heart_rate_bpm = 60 / avg_rr_interval_seconds
print(f"Average Heart Rate: {avg_heart_rate_bpm:.2f} BPM")

recording_length_seconds = len(ecg) / fs
print(f"Recording Length: {recording_length_seconds:.2f} seconds")