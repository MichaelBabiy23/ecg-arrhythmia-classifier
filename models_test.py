# Combined testing script for Decision Tree, Random Forest, and AdaBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
)

# Import your implementations
from Decision_tree import DecisionTreeClassifier as DT_ours
from RandomForestClassifier import RandomForestClassifier as RF_ours
from AdaBoostClassifier import AdaBoostClassifier as AB_ours

# 1. Load data
data = np.load('all_ecg_features.npz')
X = data['X']
y = data['y']
print(X.shape, y.shape)
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

# 6. Print class distribution
print("Class distribution in training set:")
for label, pct in (y_train.value_counts(normalize=True) * 100).items():
    print(f"  Class {label}: {pct:.2f}%")

# Helper for evaluation
def evaluate(name, y_true, y_pred_ours, y_pred_sk, labels):
    print(f"\n=== {name} Results ===")
    print(f"Accuracy (ours):  {accuracy_score(y_true, y_pred_ours):.4f}")
    print(f"Accuracy (sklearn): {accuracy_score(y_true, y_pred_sk):.4f}\n")
    print("Classification report (ours):")
    print(classification_report(y_true, y_pred_ours))
    print("Classification report (sklearn):")
    print(classification_report(y_true, y_pred_sk))

    cm_ours   = confusion_matrix(y_true, y_pred_ours, labels=labels)
    cm_sk     = confusion_matrix(y_true, y_pred_sk,   labels=labels)
    disp_ours = ConfusionMatrixDisplay(cm_ours, display_labels=labels)
    disp_sk   = ConfusionMatrixDisplay(cm_sk,   display_labels=labels)

    plt.figure(figsize=(6,5))
    disp_ours.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} — Ours")

    plt.figure(figsize=(6,5))
    disp_sk.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} — sklearn")

# List of unique labels (for confusion matrix ordering)
labels = np.unique(y_small)

# ── Decision Tree ──────────────────────────────────────────────────────────────
# sklearn stump
sk_dt = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=20,
    criterion='gini', random_state=42
)
sk_dt.fit(X_train, y_train)
y_pred_sk_dt = sk_dt.predict(X_test)

# our tree
dt_model_ours = DT_ours(max_depth=5)
dt_model_ours.fit(X_train, y_train, min_samples_split=20)
y_pred_dt_ours = dt_model_ours.predict(X_test)

evaluate("Decision Tree", y_test, y_pred_dt_ours, y_pred_sk_dt, labels)

# ── Random Forest ──────────────────────────────────────────────────────────────
# sklearn RF
sk_rf = RandomForestClassifier(
    n_estimators=100, max_depth=4,
    max_features=0.5, max_samples=0.5,
    min_samples_leaf=1, random_state=42,
    bootstrap=True
)
sk_rf.fit(X_train, y_train)
y_pred_sk_rf = sk_rf.predict(X_test)

# our RF
rf_model_ours = RF_ours(
    n_estimators=100,
    feature_percentage=0.5,
    sample_percentage=0.5,
    max_depth=4,
    min_samples_split=1
)
rf_model_ours.fit(X_train, y_train)
y_pred_rf_ours = rf_model_ours.predict(X_test)

evaluate("Random Forest", y_test, y_pred_rf_ours, y_pred_sk_rf, labels)

# ── AdaBoost ──────────────────────────────────────────────────────────────────
# sklearn AdaBoost
base = DecisionTreeClassifier(max_depth=1)
sk_ab = AdaBoostClassifier(
    estimator=base,
    n_estimators=60, learning_rate=1.0,
    algorithm="SAMME", random_state=42
)
sk_ab.fit(X_train, y_train)
y_pred_sk_ab = sk_ab.predict(X_test)

# our AdaBoost
ab_model_ours = AB_ours(max_iterations=60)
ab_model_ours.fit(X_train.values, y_train.values)
y_pred_ab_ours = ab_model_ours.predict(X_test.values)

evaluate("AdaBoost", y_test, y_pred_ab_ours, y_pred_sk_ab, labels)

# --- 1) Grouped Bar Charts of Precision/Recall/F1 for Each Model ---
models = {
    "Decision Tree": y_pred_dt_ours,
    "Random Forest": y_pred_rf_ours,
    "AdaBoost":       y_pred_ab_ours
}

# compute per-class metrics
metrics = {}
for name, y_pred in models.items():
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred,
        labels=labels,
        zero_division=0
    )
    metrics[name] = {"precision": p, "recall": r, "f1": f1}

# plot one chart for each metric type
for metric in ["precision", "recall", "f1"]:
    plt.figure(figsize=(8, 4))
    x     = np.arange(len(labels))
    width = 0.25

    for i, (model_name, vals) in enumerate(metrics.items()):
        plt.bar(x + i*width, vals[metric], width, label=model_name)

    plt.xticks(x + width, labels)
    plt.xlabel("Class")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} by Model and Class")
    plt.legend()
    plt.tight_layout()

# show all at once
plt.show()
