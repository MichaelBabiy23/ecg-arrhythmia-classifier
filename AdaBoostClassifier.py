import numpy as np
from Decision_tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.models = []
        self.alphas = []
        self.classes = None

    def fit(self, X, y):
        # Make X and y np arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        self.classes = np.unique(y)
        # Init weights
        w = np.ones(n_samples) / n_samples
        K = len(self.classes)
        for t in range(self.max_iterations):
            tree = DecisionTreeClassifier(sample_weights=w, max_depth=1)
            # Fit the tree
            tree.fit(X, y)

            # Get predictions on the full training set
            pred = tree.predict(X)

            # Find which examples it got wrong
            miss = y != pred

            # Compute weighted error
            weighted_error = np.sum(w[miss])

            if weighted_error <= 0:
                break

            # Compute level of expertise (SAMME)
            alpha_t = 1/2 * np.log((1 - weighted_error) / weighted_error) + np.log(K - 1)

            # Update weights
            # misclassified → up, correctly classified → down
            w *= np.exp(alpha_t * miss)

            # Normalize the weights
            w /= np.sum(w)

            self.models.append(tree)
            self.alphas.append(alpha_t)

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        K = len(self.classes)

        # score matrix: rows=samples, cols=classes
        S = np.zeros((n_samples, K))

        for alpha_t, model in zip(self.alphas, self.models):
            # Get predictions
            pred = model.predict(X)

            # add alpha to the predicted class column
            for i in range(n_samples):
                pred_cls_idx = self.class_to_idx(pred[i])
                S[i, pred_cls_idx] += alpha_t

        # take argmax for each sample
        best = np.argmax(S, axis=1)
        return self.classes[best]

    def class_to_idx(self, cls):
        for i in range(len(self.classes)):
            if cls == self.classes[i]:
                return i


