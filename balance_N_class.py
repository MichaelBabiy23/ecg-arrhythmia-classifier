import numpy as np
import random

# Load the existing features and labels
loaded = np.load('/Users/michaelbabiy/ecg-arrhythmia-classifier/ecg-arrhythmia-classifier/all_ecg_features.npz')
X = loaded['X']
y = loaded['y']

# Find indices of 'N' class
n_indices = [i for i, label in enumerate(y) if label == 'N']

# Randomly select 100 indices from 'N' class
random.seed(42)  # For reproducibility
selected_n_indices = random.sample(n_indices, min(100, len(n_indices)))

# Find indices of all other classes
other_indices = [i for i, label in enumerate(y) if label != 'N']

# Combine selected 'N' indices with all other class indices
final_indices = selected_n_indices + other_indices

# Filter X and y
X_final = X[final_indices]
y_final = y[final_indices]

# Save the filtered data to a new file
np.savez('all_ecg_features_balanced.npz', X=X_final, y=y_final)

print(f"Original N count: {len(n_indices)}")
print(f"New N count: {len(selected_n_indices)}")
print(f"Total samples in new file: {len(final_indices)}")
