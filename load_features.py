# Load the saved features and labels
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.load('all_ecg_features.npz')
X = data['X']  # Your features
y = data['y']  # Your labels

print("Original Features (X) shape:", X.shape)
print("Original Labels (y) shape:", y.shape)
print("First 5 original feature vectors (X):")
print(X[:5])

# 📊 Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst 5 scaled feature vectors (X_scaled):")
print(X_scaled[:5])
print("Mean of scaled features (should be close to 0):", np.mean(X_scaled))
print("Std Dev of scaled features (should be close to 1):", np.std(X_scaled))

print("Features (X) shape:", X.shape)
print("Labels (y) shape:", y.shape)
print("First 5 feature vectors (X):")
print(X[:5])
print("First 5 labels (y):")
print(y[:5]) 