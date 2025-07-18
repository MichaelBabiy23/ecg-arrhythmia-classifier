# 1️⃣ Load helpful tools
import wfdb  # ECG file reading
from wfdb.processing import xqrs_detect, ann2rr  # Heartbeat detection & RR intervals
import numpy as np  # Numerical ops
import pywt  # Wavelet transforms
import scipy.stats  # Statistical functions
import matplotlib.pyplot as plt  # For plotting ECG


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
# 'atr' refers to the annotation file extension.
ecg = record[0].flatten()  # Flatten ECG signal
fs = record[1]['fs']  # Sampling frequency

# Define the heartbeat window size early, as it's needed for plotting
window = int(0.25 * fs)  # 0.25 sec before/after spike

# 4️⃣ Find every heartbeat spike (R peak)
rpeaks = xqrs_detect(ecg, fs=fs)  # Pan–Tompkins R-peak detection
# This method uses filters and special math to pick out real heartbeats

# 📊 Plot ECG with R-peaks
plt.figure(figsize=(12, 6))

# Define the time window for plotting (e.g., first 10 seconds)
plot_duration = 10  # seconds
plot_samples = int(plot_duration * fs)

# Slice the ECG signal and R-peaks for the plot window
ec_plot = ecg[:plot_samples]
rpeaks_plot = rpeaks[rpeaks < plot_samples]

plt.plot(ec_plot)
plt.plot(rpeaks_plot, ec_plot[rpeaks_plot], 'rx', markersize=8)  # Mark R-peaks
plt.title('ECG Signal with Detected R-Peaks (First 10 Seconds)')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 📈 Plot Wavelet Decomposition for a Sample Heartbeat
# Take the first detected R-peak as an example
first_r = rpeaks[0]
first_seg = ecg[max(0, first_r - window): first_r + window]
coeffs_example = pywt.wavedec(first_seg, 'db1', level=3)

# Prepare subplot layout: 1 for original, then 3 for details, 1 for approx
num_subplots = len(coeffs_example)
plt.figure(figsize=(12, 2 * num_subplots))

# Plot the original segment for reference (optional, but good for context)
plt.subplot(num_subplots + 1, 1, 1) # +1 for the original segment
plt.plot(first_seg)
plt.title('Original Heartbeat Segment')
plt.grid(True)

plt.plot(ecg[:fs*10])  # First 10 seconds
plt.plot(rpeaks[rpeaks < fs*10], ecg[rpeaks[rpeaks < fs*10]], 'rx')
plt.title("ECG with Detected R-Peaks")

# Plot approximation and detail coefficients
# The coeffs list is [cA_n, cD_n, cD_n-1, ..., cD_1]
# cA_n is approximation at highest level, cD_i are details

for i, c in enumerate(coeffs_example):
    if i == 0:
        # First element is approximation coefficients
        title = f'Approximation Coefficients (A{num_subplots-1})'
    else:
        # Subsequent elements are detail coefficients from highest to lowest level
        title = f'Detail Coefficients (D{num_subplots-i})'

    plt.subplot(num_subplots + 1, 1, i + 2) # +2 because subplot 1 is original
    plt.plot(c)
    plt.title(title)
    plt.grid(True)

plt.tight_layout()
plt.show()

# 5️⃣ Measure time gaps between beats
# rr_intervals1 = ann2rr(record_name='physionet.org/files/mitdb/1.0.0/100', extension='atr', as_array=True)
rr_intervals = np.diff(rpeaks) / fs # in seconds
rr_intervals = np.insert(rr_intervals, 0, rr_intervals[0]) # Adds the first RR value at index 0 to fix the shape

# 6️⃣ Prepare lists to save data for each beat
beat_features = []
beat_labels = []

# 7️⃣ For each heartbeat, grab info
for i, r in enumerate(rpeaks):
    seg = ecg[max(0, r-window): r+window]  # Heartbeat window
    feats = []
    # ➕ Basic stats: length, avg height, wiggle, top-to-bottom
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
avg_rr_interval_seconds = np.mean(rr_intervals)
avg_heart_rate_bpm = 60 / avg_rr_interval_seconds
print(f"Average Heart Rate: {avg_heart_rate_bpm:.2f} BPM")

recording_length_seconds = len(ecg) / fs
print(f"Recording Length: {recording_length_seconds:.2f} seconds")