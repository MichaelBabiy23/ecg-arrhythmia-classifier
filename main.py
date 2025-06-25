from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import Decision_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
y_df = pd.DataFrame(y, columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.33, random_state=42, stratify=y)

decision_tree = Decision_tree.build_decision_tree(X_train, y_train, 10)

y_pred = []

for sample in X_test:
    y_pred.append(decision_tree.predict(sample))

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics (per class: precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=y.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()





