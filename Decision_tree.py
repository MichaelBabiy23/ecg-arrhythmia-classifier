import numpy as np
import pandas as pd
import TreeNode
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth, X=np.array([]), y=np.array([]), sample_weights=None):
        self.sample_weights = sample_weights
        self.X=X
        self.y=y
        self.max_depth = max_depth
        self.node = None
        self.classes = None

    # Calculate gini impurity
    def gini_cost_function(self, y, current_weights):
        if len(y) == 0:
            return 0
        total_weight = np.sum(current_weights)

        probabilities = np.zeros(len(self.classes))
        for sample in range(len(current_weights)):
            current_class = y[sample]
            current_cls_idx = self.class_to_idx(current_class)
            probabilities[current_cls_idx] += current_weights[sample]

        probabilities = probabilities / total_weight
        return 2 * np.sum(probabilities * (1 - probabilities))

    # Find best split
    def find_best_split(self, indices):
        n_features = self.X.shape[1]
        n_samples = len(indices)
        if n_samples <= 1:
            return None, None, None, None

        # Current data
        current_X = self.X[indices]
        current_y = self.y[indices]
        current_weights = self.sample_weights[indices]
        parent_gini = self.gini_cost_function(current_y, current_weights)

        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = current_X[:, feature_idx]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            # Try thresholds between consecutive unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                # Split based on threshold
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate weighted gini for this split
                left_gini = self.gini_cost_function(current_y[left_mask], current_weights[left_mask])
                right_gini = self.gini_cost_function(current_y[right_mask], current_weights[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = indices[left_mask]
                    best_right_indices = indices[right_mask]

        return best_feature, best_threshold, best_left_indices, best_right_indices


    # Build decision tree
    def build_decision_tree(self, indices=None, current_depth=0, min_samples_split=20):
        if indices is None:
            indices = np.arange(len(self.X))

        current_y = self.y[indices]
        n_samples = len(indices)

        # Stopping conditions
        if (current_depth >= self.max_depth or
                len(np.unique(current_y)) == 1 or
                n_samples <= min_samples_split):
            # Create leaf node
            most_common = Counter(current_y).most_common(1)[0][0]
            return TreeNode.TreeNode(pred=most_common)

        # Find best split
        best_feature, best_threshold, left_indices, right_indices = self.find_best_split(indices)

        if best_feature is None or left_indices is None or right_indices is None:
            # No good split found, create leaf
            most_common = Counter(current_y).most_common(1)[0][0]
            return TreeNode.TreeNode(pred=most_common)

        # Create internal node
        node = TreeNode.TreeNode(
            threshold=best_threshold,
            feature_index=best_feature,
            pred=Counter(current_y).most_common(1)[0][0]  # Fallback prediction
        )

        # Recursively build children
        node.left = self.build_decision_tree(
            left_indices, current_depth + 1, min_samples_split
        )
        node.right = self.build_decision_tree(
            right_indices, current_depth + 1, min_samples_split
        )

        return node


    def predict(self, X):
        """Predict for multiple samples at once"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for i in range(len(X)):
            pred = self.predict_single(X[i], self.node)
            predictions.append(pred)

        return np.array(predictions)


    def predict_single(self, sample, node):
        """Predict for a single sample"""
        # If leaf node (no childrens)
        if not node.right and not node.left:
            return node.pred

        # Traverse tree
        if sample[node.feature_index] <= node.threshold:
            if hasattr(node, 'left') and node.left is not None:
                return self.predict_single(sample, node.left)
            else:
                return node.pred
        else:
            if hasattr(node, 'right') and node.right is not None:
                return self.predict_single(sample, node.right)
            else:
                return node.pred


    # Wrapper function to convert pandas to numpy
    def fit(self, X_df, y_series, min_samples_split=20):
        """Convert pandas input to numpy and build tree"""
        X_np = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        y_np = y_series.values if isinstance(y_series, pd.Series) else y_series

        self.X = X_np
        self.y = y_np

        if self.sample_weights is None:
            self.sample_weights = np.ones(len(y_series))

        self.classes = np.unique(y_np)

        self.node = self.build_decision_tree(
                min_samples_split=min_samples_split
        )

    def class_to_idx(self, cls):
        for i in range(len(self.classes)):
            if cls == self.classes[i]:
                return i