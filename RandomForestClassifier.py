import random
from Decision_tree import DecisionTreeClassifier
import statistics
import numpy as np


class RandomForestClassifier:

    def __init__(
            self, min_samples_split=20, n_estimators=10, 
            feature_percentage=0.5, sample_percentage=0.5, max_depth=5
            ):
        # Number of trees in the forest
        self.n_estimators = n_estimators
        # Minimum samples required to split a node in each tree
        self.min_samples_split = min_samples_split
        # Maximum depth of each tree
        self.max_depth = max_depth
        # List to store (tree, feature_indices) tuples
        self.trees = []
        # Fraction of features to use for each tree
        self.feature_percentage = feature_percentage
        # Fraction of samples to use for each tree (bootstrap)
        self.sample_percentage = sample_percentage

        # Sanity checks for percentages
        if not sample_percentage or sample_percentage <= 0 or sample_percentage > 1:
            self.sample_percentage = 0.5

        if not feature_percentage or feature_percentage <= 0 or feature_percentage > 1:
            self.feature_percentage = 0.5

    def select_features(self, X):
        # Select a random subset of features (columns) for a tree
        n_features = int(self.feature_percentage * X.shape[1])
        feature_indices = random.sample(range(X.shape[1]), n_features)
        return X[:, feature_indices], feature_indices

    def select_sample(self, X, y):
        # Create a bootstrap sample (random rows with replacement)
        n_samples = int(self.sample_percentage * X.shape[0])
        n_samples = max(1, n_samples)  # Ensure at least 1 sample
        sample_indices = np.random.choice(X.shape[0], size=n_samples, replace=True)
        return X[sample_indices], y[sample_indices]

    def fit(self, X, y):
        # Train the random forest on the data
        X = np.asarray(X)
        y = np.asarray(y)

        for _ in range(self.n_estimators):
            # Bootstrap sampling: select random rows
            sample_X, sample_y = self.select_sample(X, y)
            # Feature selection: select random columns
            filtered_X, feature_indices = self.select_features(sample_X)
            # Train a decision tree on the sampled data
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(filtered_X, sample_y, min_samples_split=self.min_samples_split)
            # Store the tree and its feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # Predict class labels for each sample in X using majority vote
        X = np.asarray(X)
        n_samples = X.shape[0]
        final_preds = []

        # For each sample...
        for i in range(n_samples):
            votes = []
            # Ask every tree for its prediction on that sample
            for tree, feature_indices in self.trees:
                # Select the same features as used for this tree
                x_sub = X[i, feature_indices].reshape(1, -1)
                votes.append(tree.predict(x_sub)[0])
            # Use majority vote across all trees
            final_preds.append(statistics.mode(votes))

        return np.array(final_preds)

