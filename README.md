# ecg-arrhythmia-classifier

Visualize waveforms
https://www.physionet.org/lightwave/?db=mitdb/1.0.0 


For the dataset:
'
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ DESTINATION
'

ECG
https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/SinusRhythmLabels.svg/1024px-SinusRhythmLabels.svg.png

Basic Statistics (4 features):
len(seg): The number of data points in the heartbeat segment.
np.mean(seg): The average amplitude of the heartbeat segment.
np.std(seg): The standard deviation, indicating the variability or "wiggle" of the segment.
np.max(seg) - np.min(seg): The peak-to-peak amplitude (range) of the segment.
Timing Information (2 features):
rr_intervals[i]: The time elapsed between the current R-peak and the next R-peak.
rr_intervals[i-1] if i > 0 else rr_intervals[i]: The time elapsed between the previous R-peak and the current R-peak (or the current RR interval if it's the first beat).
Wavelet Coefficients Statistics (8 features):
The pywt.wavedec(seg, 'db1', level=3) function performs a 3-level discrete wavelet transform. This produces 4 arrays of coefficients (one approximation coefficient array and three detail coefficient arrays).
For each of these 4 coefficient arrays, we calculate its np.mean() (average) and np.std() (standard deviation).
So, 4 arrays * 2 stats/array = 8 features.
Morphological Statistics (2 features):
scipy.stats.skew(seg): Measures the asymmetry of the heartbeat segment's amplitude distribution (how "tilted" it is).
scipy.stats.kurtosis(seg): Measures the "peakiness" or "flatness" of the heartbeat segment's amplitude distribution.
Adding these up: 4 (basic) + 2 (timing) + 8 (wavelet) + 2 (morphological) = 16 features in total for each heartbeat.