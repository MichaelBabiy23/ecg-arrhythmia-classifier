import random

import pandas as pd

from Decision_tree import DecisionTreeClassifier
import statistics
import numpy as np

class RandomForestClassifier:
    def __init__(self, min_samples_split=20, n_estimators=10, feature_percentage=0.5, sample_percentage=0.5, max_depth=5):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self.feature_percentage = feature_percentage
        self.sample_percentage = sample_percentage

        if not sample_percentage or sample_percentage <= 0 or sample_percentage > 1:
            self.sample_percentage = 0.5

        if not feature_percentage or feature_percentage <= 0 or feature_percentage > 1:
            self.feature_percentage = 0.5

    def select_features(self, X):
        # Select random subset of features
        n_features = int(self.feature_percentage * X.shape[1])

        feature_indices = random.sample(range(X.shape[1]), n_features)
        return X[:, feature_indices], feature_indices

    def select_sample(self, X, y):
        # Create bootstrap sample with replacement
        n_samples = int(self.sample_percentage * X.shape[0])
        n_samples = max(1, n_samples)  # Ensure at least 1 sample

        # Bootstrap sampling (with replacement)
        sample_indices = np.random.choice(X.shape[0], size=n_samples, replace=True)

        return X[sample_indices], y[sample_indices]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            sample_X, sample_y = self.select_sample(X, y)

            # Feature selection
            filtered_X, feature_indices = self.select_features(sample_X)

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(filtered_X, sample_y, min_samples_split=self.min_samples_split)
            self.trees.append((tree, feature_indices))

            # 🔹 Print class distribution as percentages
            class_dist = sample_y.value_counts(normalize=True) * 100
            print(f"Tree {i + 1}: Class distribution in sample:")
            for label, percent in class_dist.items():
                print(f"  Class {label}: {percent:.2f}%")

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        final_preds = []

        # for each sample…
        for i in range(n_samples):
            votes = []
            # ask every tree for its prediction on that sample
            for tree, feature_indices in self.trees:
                # features should be a list/array of column indices
                x_sub = X[i, feature_indices].reshape(1, -1)
                votes.append(tree.predict(x_sub)[0])
            # majority vote
            final_preds.append(statistics.mode(votes)[0])

        return np.array(final_preds)









