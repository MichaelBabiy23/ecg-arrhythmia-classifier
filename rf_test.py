import RandomForestClassifier as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


data = np.load('all_ecg_features.npz')
X = data['X']
y = data['y']

feature_names = [
    'length', 'mean', 'std', 'range',
    'rr_interval_current', 'rr_interval_prev',
]

# Add wavelet feature names (db1, level 3 → 4 sets of coeffs)
for i in range(4):
    feature_names.append(f'wavelet_L{i}_mean')
    feature_names.append(f'wavelet_L{i}_std')

feature_names.extend(['skewness', 'kurtosis'])
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y)

X_small = X_df.sample(n=500, random_state=42)
y_small = y_series.loc[X_small.index]

X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42, stratify=y_small)

# 🔹 Print class distribution as percentages
class_dist = y_train.value_counts(normalize=True) * 100
print(f"Class distribution in original sample:")
for label, percent in class_dist.items():
    print(f"  Class {label}: {percent:.2f}%")

random_forest = rf.RandomForestClassifier(n_estimators=20, feature_percentage=1, sample_percentage=1, max_depth=4)
random_forest.fit(X_train, y_train)

y_pred = X_test.apply(lambda row: random_forest.predict(row), axis=1)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics (per class: precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=y_small.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_small.unique())

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

