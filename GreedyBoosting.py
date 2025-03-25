import numpy as np
from DecisionStump import DecisionStump


class GreedyBoosting:
    """
    The class that takes the role of the model in decision stump.
    """
    def __init__(self, num_classes, num_stumps, num_thresholds):
        self.num_classes = num_classes
        self.num_stumps = num_stumps
        self.num_thresholds = num_thresholds
        self.stumps = []

    def fit(self, train_features, train_labels):
        """
        A function that trains the decision stump with a training set
        Args:
            train_features: A numpy array of shape (n_samples, n_features)
            train_labels: A numpy array of shape (n_samples,)

        Returns: None
        """
        n_samples = train_features.shape[0]
        scores = np.zeros((n_samples, self.num_classes))

        for _ in range(self.num_stumps):
            stump = DecisionStump(num_classes=self.num_classes)
            stump.fit(train_features, train_labels, scores, num_thresholds=self.num_thresholds)
            scores = stump.predict(train_features, scores)
            self.stumps.append(stump)

    def predict(self, test_features):
        """
        A function that predicts labels based on the training set
        Args:
            test_features:

        Returns: A numpy array of shape (n_samples,)
        """

        # Initializing the scores to an array of zeros
        scores = np.zeros((test_features.shape[0], self.num_classes))
        for stump in self.stumps:
            scores = stump.predict(test_features, scores)
        return np.argmax(scores, axis=1)

    def predict_with_stumps(self, test_features, num_stumps):
        scores = np.zeros((test_features.shape[0], self.num_classes))
        for stump in self.stumps[:num_stumps]:
            scores = stump.predict(test_features, scores)
        return np.argmax(scores, axis=1)

