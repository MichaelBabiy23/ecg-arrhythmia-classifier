import random
from Decision_tree import predict_decision_tree, build_decision_tree
import statistics


class RandomForestClassifier:
    def __init__(self, n_estimators=10, feature_percentage=0.5, sample_percentage=0.5, max_depth=5):
        self.n_estimators = n_estimators
        self.feature_percentage = feature_percentage
        self.sample_percentage = sample_percentage
        self.max_depth = max_depth
        self.trees = []


    def filter_data(self, X, y):
        feature_idx = random.sample(range(X.shape[1]), int(self.feature_percentage * X.shape[1]))
        feature_filtered_X = X.iloc[:, feature_idx]

        sample_idx = random.sample(range(X.shape[0]), int(self.sample_percentage * X.shape[0]))
        filtered_X = feature_filtered_X.iloc[sample_idx, :]
        filtered_y = y.iloc[sample_idx]

        return filtered_X, filtered_y


    def fit(self, X, y):
        for i in range(self.n_estimators):
            filtered_X, filtered_y = self.filter_data(X, y)
            self.trees.append((build_decision_tree(filtered_X, filtered_y, self.max_depth), filtered_X.columns))

            # 🔹 Print class distribution as percentages
            class_dist = filtered_y.value_counts(normalize=True) * 100
            print(f"Tree {i + 1}: Class distribution in sample:")
            for label, percent in class_dist.items():
                print(f"  Class {label}: {percent:.2f}%")

    def predict(self, new_sample):
        preds = []
        for tree, features in self.trees:
            sample_subset = new_sample[features]
            preds.append(predict_decision_tree(sample_subset, tree))

        print(preds.count('N'))
        return statistics.mode(preds)[0]









