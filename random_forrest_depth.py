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


# min_sections = [2, 5, 10, 20]
# accuracies = []
# for min_samples in min_sections:
#     print(f"Evaluating min_samples_split: {min_samples}")
#     dt_model_ours = DT_ours(max_depth=5)
#     dt_model_ours.fit(X_train, y_train, min_samples_split=min_samples)
#     y_pred_dt_ours = dt_model_ours.predict(X_test)
#     acc = accuracy_score(y_test, y_pred_dt_ours)
#     accuracies.append(acc)

#     # --- Confusion Matrix ---
#     cm = confusion_matrix(y_test, y_pred_dt_ours, labels=labels)
#     disp = ConfusionMatrixDisplay(cm, display_labels=labels)
#     plt.figure(figsize=(6,5))
#     disp.plot(cmap='Blues', values_format='d')
#     plt.title(f"DT Confusion Matrix (min_samples_split={min_samples})")

#     # --- Precision, Recall, F1 ---
#     from sklearn.metrics import precision_recall_fscore_support
#     p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_dt_ours, labels=labels, zero_division=0)
#     print(f"Precision: {p}")
#     print(f"Recall:    {r}")
#     print(f"F1-score:  {f1}")
#     # Bar plot for metrics
#     x = np.arange(len(labels))
#     width = 0.25
#     plt.figure(figsize=(8, 4))
#     plt.bar(x, p, width, label='Precision')
#     plt.bar(x + width, r, width, label='Recall')
#     plt.bar(x + 2*width, f1, width, label='F1-score')
#     plt.xticks(x + width, labels)
#     plt.xlabel('Class')
#     plt.ylabel('Score')
#     plt.title(f"Precision/Recall/F1 (min_samples_split={min_samples})")
#     plt.legend()
#     plt.tight_layout()

# #Plot min_samples_split vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(min_sections, accuracies, marker='o')
# plt.xlabel('Min Samples Split')
# plt.ylabel('Accuracy')
# plt.title('Min Samples Split vs Accuracy')
# plt.grid(True)
# plt.show()


# Collect accuracy for each depth
depths = list(range(1, 11))
accuracies = []
for depth in depths:
    print(f"Evaluating depth: {depth}")
    dt_model_ours = DT_ours(max_depth=depth)
    dt_model_ours.fit(X_train, y_train, min_samples_split=2)
    y_pred_dt_ours = dt_model_ours.predict(X_test)
    acc = accuracy_score(y_test, y_pred_dt_ours)
    accuracies.append(acc)

        # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred_dt_ours, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    plt.figure(figsize=(6,5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"DT Confusion Matrix (depth={depth})")

    # --- Precision, Recall, F1 ---
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_dt_ours, labels=labels, zero_division=0)
    print(f"Precision: {p}")
    print(f"Recall:    {r}")
    print(f"F1-score:  {f1}")
    # Bar plot for metrics
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x, p, width, label='Precision')
    plt.bar(x + width, r, width, label='Recall')
    plt.bar(x + 2*width, f1, width, label='F1-score')
    plt.xticks(x + width, labels)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f"Precision/Recall/F1 (depth={depth})")
    plt.legend()
    plt.tight_layout()

# Plot depth vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(depths, accuracies, marker='o')
plt.xlabel('Decision Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.grid(True)
plt.show()

# # Show evaluation for the last depth
# # Test different n_estimators for your Random Forest
# n_estimators_list = [1, 5, 10, 20, 50, 100]
# rf_accuracies = []

# for n_estimators in n_estimators_list:
#     print(f"Evaluating n_estimators: {n_estimators}")
#     rf_model_ours = RF_ours(n_estimators=n_estimators, max_depth=5)  # or pick depth you like
#     rf_model_ours.fit(X_train, y_train)
#     y_pred_rf_ours = rf_model_ours.predict(X_test)
#     acc = accuracy_score(y_test, y_pred_rf_ours)
#     rf_accuracies.append(acc)

# # Plot n_estimators vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(n_estimators_list, rf_accuracies, marker='o')
# plt.xlabel('Number of Trees (n_estimators)')
# plt.ylabel('Accuracy')
# plt.title('Random Forest — n_estimators vs Accuracy')
# plt.grid(True)
# plt.show()


# n_features_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
# rf_accuracies = []

# for n_features in n_features_percentages:
#     print(f"Evaluating n_features: {n_features}")
#     rf_model_ours = RF_ours(n_estimators=100, max_depth=5, max_features=n_features)  # or pick depth you like
#     rf_model_ours.fit(X_train, y_train)
#     y_pred_rf_ours = rf_model_ours.predict(X_test)
#     acc = accuracy_score(y_test, y_pred_rf_ours)
#     rf_accuracies.append(acc)

# # Plot n_features vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(n_features_percentages, rf_accuracies, marker='o')
# plt.xlabel('Feature Percentage (max_features)')
# plt.ylabel('Accuracy')
# plt.title('Random Forest — Feature Percentage vs Accuracy')
# plt.grid(True)
# plt.show()


# n_features_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
# rf_accuracies = []

# for n_features in n_features_percentages:
#     print(f"Evaluating n_features: {n_features}")
#     rf_model_ours = RF_ours(n_estimators=20, max_depth=5, feature_percentage=n_features)  # or pick depth you like
#     rf_model_ours.fit(X_train, y_train)
#     y_pred_rf_ours = rf_model_ours.predict(X_test)
#     acc = accuracy_score(y_test, y_pred_rf_ours)
#     rf_accuracies.append(acc)

# # Plot n_features vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(n_features_percentages, rf_accuracies, marker='o')
# plt.xlabel('Feature Percentage (max_features)')
# plt.ylabel('Accuracy')
# plt.title('Random Forest — Feature Percentage vs Accuracy')
# plt.grid(True)
# plt.show()


# n_fsamples_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
# rf_accuracies = []

# for n_fsamples in n_fsamples_percentages:
#     print(f"Evaluating n_fsamples: {n_fsamples}")
#     rf_model_ours = RF_ours(n_estimators=20, max_depth=5, feature_percentage=0.6, sample_percentage=n_fsamples)  # or pick depth you like
#     rf_model_ours.fit(X_train, y_train)
#     y_pred_rf_ours = rf_model_ours.predict(X_test)
#     acc = accuracy_score(y_test, y_pred_rf_ours)
#     rf_accuracies.append(acc)

# # Plot n_fsamples vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(n_fsamples_percentages, rf_accuracies, marker='o')
# plt.xlabel('Sample Percentage (max_samples)')
# plt.ylabel('Accuracy')
# plt.title('Random Forest — Sample Percentage vs Accuracy')
# plt.grid(True)
# plt.show()

