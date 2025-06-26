from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

from Decision_tree import predict_decision_tree

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


decision_tree_b = DecisionTreeClassifier(
    max_depth=5,                  # Maximum depth of the tree
    min_samples_leaf=20,           # Minimum samples required at a leaf node
    criterion='gini',             # Split criterion ('gini' or 'entropy')
    random_state=42
)
print('training theirs...')
# Train the model
decision_tree_b.fit(X_train, y_train)

# Make predictions
y_pred_b = decision_tree_b.predict(X_test)
print('now ours...')
decision_tree = Decision_tree.build_decision_tree(X_train, y_train, 5)

y_pred = X_test.apply(lambda row: predict_decision_tree(row, decision_tree), axis=1)

# Accuracy
print("Accuracy ours:", accuracy_score(y_test, y_pred), 'Accuracy theirs:', accuracy_score(y_test, y_pred_b))

# Detailed metrics (per class: precision, recall, f1-score)
print("\nClassification Report ours:")
print(classification_report(y_test, y_pred))

print("\nClassification Report theirs:")
print(classification_report(y_test, y_pred_b))

cm = confusion_matrix(y_test, y_pred, labels=y_small.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_small.unique())

cm_b = confusion_matrix(y_test, y_pred_b, labels=y_small.unique())
disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=y_small.unique())

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix ours")

plt.figure(figsize=(8, 6))
disp_b.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix theirs")

plt.show()




