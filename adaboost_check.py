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
from AdaBoostClassifier import AdaBoostClassifier as AB_ours


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
X_small = X_df.sample(n=300, random_state=42)
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

# our AdaBoost
ab_model_ours = AB_ours(max_iterations=2000)
ab_model_ours.fit(X_train.values, y_train.values)
y_pred_ab_ours = ab_model_ours.predict(X_test.values)

evaluate("AdaBoost", y_test, y_pred_ab_ours, labels)

# Show all confusion matrices
plt.show()