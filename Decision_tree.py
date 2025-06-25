import numpy as np
import pandas as pd
import TreeNode
import statistics


# Calculate a section cost
def gini_cost_function(relative_y):
    classes = relative_y.unique()
    cost = 0
    for c in classes:
        prob = np.mean(relative_y == c)
        cost += prob * (1 - prob)

    return 2*cost

# Calculate total split improvement
def find_improvement(father_cost, left_indexes, right_indexes, relative_y):
    samples_in_left_section = len(left_indexes)
    samples_in_right_section = len(right_indexes)
    left_child_cost = gini_cost_function(relative_y[left_indexes])
    right_child_cost = gini_cost_function(relative_y[right_indexes])
    weighted_avg = (left_child_cost * samples_in_left_section) + (samples_in_right_section * right_child_cost)
    weighted_avg /= (samples_in_left_section  + samples_in_right_section)
    return  father_cost - weighted_avg

# Create a split in the data
def create_split(relative_X, feature, threshold):
    return relative_X[relative_X[feature] <= threshold].index(), relative_X[relative_X[feature] >= threshold].index()

# Check for every threshold
def find_best_split(relative_X, relative_y, father_cost):


    def get_candidate_thresholds(values):
        # Assumes values are sorted and unique
        return [(v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:])]


    features = relative_X.shape[1]
    best_improvement = -1
    best_split_feature_index = 0
    best_split_threshold = relative_X[0,0]
    best_left_indexes = None
    best_right_indexes = None

    for feature_index in features:
        values = relative_X.loc[relative_X, feature_index].sort_values().unique()
        thresholds = get_candidate_thresholds(values)
        for threshold in thresholds:
            left_index, right_index = create_split(relative_X, feature_index, threshold)
            improvement = find_improvement(father_cost, left_indexes=left_index,
                                           right_indexes=right_index, relative_y=relative_y)
            if improvement > best_improvement:
                best_improvement = improvement
                best_split_threshold = threshold
                best_split_feature_index = feature_index
                best_left_indexes = left_index
                best_right_indexes = right_index

    return best_split_feature_index, best_split_threshold, best_left_indexes, best_right_indexes

# Build decision tree
def build_decision_tree(relative_X, relative_y, max_depth, current_depth=0):
    if current_depth >= max_depth:
        return None

    cost = gini_cost_function(relative_y)
    best_split_feature, best_split_threshold, best_left_indexes, best_right_indexes = find_best_split(relative_X, relative_y, cost)

    node = TreeNode.TreeNode(cost=cost, threshold=best_split_threshold, feature_index=best_split_feature)

    if len(best_left_indexes) > 20 or len(relative_y[best_left_indexes].unique()) > 1:
        node.left = build_decision_tree(relative_X[best_left_indexes], relative_y[best_left_indexes], max_depth, current_depth+1)
    if len(best_right_indexes) > 20 or len(relative_y[best_right_indexes].unique()) > 1:
        node.right = build_decision_tree(relative_X[best_right_indexes], relative_y[best_right_indexes], max_depth, current_depth+1)

    if not node.left and not node.right:
        node.pred = statistics.mode(relative_y)[0]

    return node

# Predict new sample class with decision tree
def predict_decision_tree(new_sample, decision_tree):
    if not decision_tree.left and not decision_tree.right:
        return decision_tree.pred

    if decision_tree.left and new_sample[decision_tree.feature] <= decision_tree.threshold:
        return predict_decision_tree(new_sample, decision_tree.left)

    return predict_decision_tree(new_sample, decision_tree.right)

