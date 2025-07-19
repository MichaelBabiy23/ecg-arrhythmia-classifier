import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Import your implementations
from Decision_tree import DecisionTreeClassifier as DT_ours
from RandomForestClassifier import RandomForestClassifier as RF_ours


data = np.load('all_ecg_features.npz')
X = data['X']
y = data['y']

# 2. Build feature names
feature_names = [
    'length', 'mean', 'std', 'range',
    'rr_interval_current', 'rr_interval_prev',
]
for i in range(4):
    feature_names += [f'wavelet_L{i}_mean', f'wavelet_L{i}_std']
feature_names += ['skewness', 'kurtosis']

# 3. Wrap into DataFrame/Series
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y)

# 4. Subsample to 500 for faster testing
X_small = X_df.sample(n=500, random_state=42)
y_small = y_series.loc[X_small.index]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2,
    random_state=42, stratify=y_small
)


def evaluate(name, y_true, y_pred_ours, labels):
    print(f"\n=== {name} Results ===")
    print(f"Accuracy (ours):  {accuracy_score(y_true, y_pred_ours):.4f}")
    print("Classification report (ours):")
    print(classification_report(y_true, y_pred_ours))

    cm_ours   = confusion_matrix(y_true, y_pred_ours, labels=labels)
    disp_ours = ConfusionMatrixDisplay(cm_ours, display_labels=labels)

    plt.figure(figsize=(6,5))
    disp_ours.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} — Ours")

    plt.figure(figsize=(6,5))
    plt.title(f"{name} — sklearn")


labels = np.unique(y_small)


# Collect accuracy for each depth
depths = list(range(1, 11))
accuracies = []
for depth in depths:
    print(f"Evaluating depth: {depth}")
    dt_model_ours = DT_ours(max_depth=depth)
    dt_model_ours.fit(X_train, y_train, min_samples_split=20)
    y_pred_dt_ours = dt_model_ours.predict(X_test)
    acc = accuracy_score(y_test, y_pred_dt_ours)
    accuracies.append(acc)

# Plot depth vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(depths, accuracies, marker='o')
plt.xlabel('Decision Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.grid(True)
plt.show()

# Show evaluation for the last depth
# Test different n_estimators for your Random Forest
n_estimators_list = [1, 5, 10, 20, 50, 100]
rf_accuracies = []

for n_estimators in n_estimators_list:
    print(f"Evaluating n_estimators: {n_estimators}")
    rf_model_ours = RF_ours(n_estimators=n_estimators, max_depth=5)  # or pick depth you like
    rf_model_ours.fit(X_train, y_train)
    y_pred_rf_ours = rf_model_ours.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf_ours)
    rf_accuracies.append(acc)

# Plot n_estimators vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(n_estimators_list, rf_accuracies, marker='o')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Random Forest — n_estimators vs Accuracy')
plt.grid(True)
plt.show()

