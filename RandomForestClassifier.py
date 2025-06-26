import random
from Decision_tree import predict_decision_tree, build_decision_tree
import statistics
import numpy as np

class RandomForestClassifier:
    def __init__(self, n_estimators=10, feature_percentage=0.5, sample_percentage=0.5, max_depth=5):
        self.n_estimators = n_estimators
        self.feature_percentage = feature_percentage
        self.sample_percentage = sample_percentage
        self.max_depth = max_depth
        self.trees = []

    def select_features(self, X):
        # Select random subset of features
        n_features = int(self.feature_percentage * X.shape[1])
        n_features = max(1, n_features)  # Ensure at least 1 feature
        n_features = min(n_features, X.shape[1])  # Don't exceed available features

        feature_indices = random.sample(range(X.shape[1]), n_features)
        return X.iloc[:, feature_indices]

    def bootstrap_sample(self, X, y):
        # Create bootstrap sample with replacement
        n_samples = int(self.sample_percentage * X.shape[0])
        n_samples = max(1, n_samples)  # Ensure at least 1 sample

        # Bootstrap sampling (with replacement)
        sample_indices = np.random.choice(X.shape[0], size=n_samples, replace=True)

        return X.iloc[sample_indices], y.iloc[sample_indices]


    def fit(self, X, y):
        for i in range(self.n_estimators):
            # Bootstrap sampling
            sample_X, sample_y = self.bootstrap_sample(X, y)

            # Feature selection
            filtered_X = self.select_features(sample_X)

            tree = build_decision_tree(filtered_X, sample_y, self.max_depth)
            self.trees.append((tree, filtered_X.columns))

            # 🔹 Print class distribution as percentages
            class_dist = sample_y.value_counts(normalize=True) * 100
            print(f"Tree {i + 1}: Class distribution in sample:")
            for label, percent in class_dist.items():
                print(f"  Class {label}: {percent:.2f}%")

    def predict(self, new_sample):
        preds = []
        for tree, features in self.trees:
            sample_subset = new_sample[features]
            preds.append(predict_decision_tree(sample_subset, tree))

        return statistics.mode(preds)[0]









