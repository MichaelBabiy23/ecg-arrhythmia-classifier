# ecg-arrhythmia-classifier

Visualize waveforms
https://www.physionet.org/lightwave/?db=mitdb/1.0.0 


For the dataset:
'
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ DESTINATION
'


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

The Main Melody Track (Approximation Coefficients - A3):
This is like the slow, smooth, main tune of your heartbeat song. If you just listened to this track, you'd get the general idea of the song, but it would sound a bit blurry, like a picture that's a little out of focus. It's the big, slow changes in your heartbeat.
The Fast Beat Tracks (Detail Coefficients - D3, D2, D1):
These are the "secret" ingredients that add all the exciting details and quick changes to your heartbeat song!
D3 (the coarsest detail): This track has the slightly faster beats and movements.
D2 (the mid detail): This track has even faster, more noticeable little wiggles and sudden changes.
D1 (the finest detail): This track has the super-fast, tiny "clicks" and "pops" – the quickest wiggles and all the little rapid parts that make your heartbeat unique (and sometimes noisy!).
Why do we need all these tracks?
Because sometimes, if there's a problem with your heart, it might change only a tiny, fast "wiggle" (which you'd hear on a D1 track) or a slightly slower "thump" (which you'd hear on a D3 track), even if the main melody (A3) sounds okay.
By looking at all these separate ingredient tracks, we can find tiny clues that tell us if your heart song is healthy or if it's playing a bit out of tune! It helps doctors and computers understand every little part of your heartbeat, not just the loudest "bang."

Skewness (the "Tilt-o-Meter"): (scipy.stats.skew(seg))
Imagine your hill. Is it perfectly even, like a mountain drawn by a kid? Or is one side longer or more stretched out than the other?
The "Tilt-o-Meter" tells us if your heartbeat hill is leaning to one side!
If it's perfectly balanced, the "Tilt-o-Meter" shows zero.
If the hill has a long, gentle slope on the right side, it's "leaning right" (positive tilt).
If it has a long, gentle slope on the left side, it's "leaning left" (negative tilt).
This is important because some heart problems can make your heartbeat wiggle look "tilted" in a special way!
Kurtosis (the "Peakiness-o-Meter"): (scipy.stats.kurtosis(seg))
Now, look at the very top of your heartbeat hill. Is it super pointy, like a really sharp church steeple? Or is it more flat and rounded, like a smooth, rolling hill?
The "Peakiness-o-Meter" tells us how pointy or flat the very top of your heartbeat hill is!
A high number means it's super pointy and has very long "tails" (meaning the hill goes very low on the sides).
A low number means it's more rounded or even flat on top.
Just like the tilt, the "peakiness" can also change when there's a problem with your heart's electricity, giving us another clue!
So, these two tools help us describe the exact shape of each heartbeat wiggle, adding even more secret clues to our collection to help tell if your heart song is playing just right!