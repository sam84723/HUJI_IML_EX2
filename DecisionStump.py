import numpy as np

class DecisionStump:
    """
    A decision stamps node used in greedy boosting.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.feature_index = None
        self.threshold = None
        self.direction = None # ???
        self.left_class = None # ???

    def __generate_thresholds(self, train_features, feature, num_thresholds):
        """
        Generate thresholds for decision stump.
        Args:
            train_features: A numpy array containing training features.
            feature: The index of the feature to create thresholds for
            num_thresholds: Number of thresholds to generate.

        Returns: A list of thresholds.

        """
        feature_min = train_features[:, feature].min()
        feature_max =  train_features[:, feature].max()
        return np.linspace(feature_min, feature_max, num=num_thresholds)

    def __decide_side(self, X, feature, threshold, direction):
        if direction == 'left':
            return X[:, feature] <= threshold
        else:
            return X[:, feature] > threshold

    def __evaluate_split(self, train_features, train_labels, scores, feature, threshold, direction, class_label):
        side = self.__decide_side(train_features, feature, threshold, direction)

        score_diff = np.zeros_like(scores)
        # if the side is 1, it's the left side otherwise its right side
        score_diff[side, class_label] = 1
        # The right side will receive uniform distribution
        score_diff[~side, :] = 1 / self.num_classes

        updated_scores = scores + score_diff
        predictions = np.argmax(updated_scores, axis=1)

        accuracy = np.mean(predictions == train_labels)
        return accuracy, updated_scores



    def fit(self, train_features, train_labels, scores, num_thresholds):
        """
        A function that trains the decision stump on the decision features.
        Args:
            train_features: A numpy array of shape (num_samples, num_features).
            train_labels: A numpy array of shape (num_samples,).
            scores: The scores returned by a decision tree classifier.
            num_thresholds: The number of decision thresholds to use.

        Returns: None

        """
        num_samples, num_features = train_features.shape
        best_accuracy = 0

        # We go over all the possibilities of putting a line to get the best accuracy possible
        for feature in range(num_features):
            thresholds = self.__generate_thresholds(train_features, feature, num_thresholds)

            for threshold in thresholds:
                for class_label in range(self.num_classes):
                    for direction in ['left', 'right']:
                        accuracy, updated_scores = self.__evaluate_split(train_features, train_labels, scores, feature, threshold, direction,
                                                                        class_label)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            self.feature_index = feature
                            self.threshold = threshold
                            self.direction = direction
                            self.left_class = class_label
                            self.best_scores = updated_scores


    def predict(self, test_features, scores):
        mask = self.__decide_side(test_features, self.feature_index, self.threshold, self.direction)
        score_diff = np.zeros_like(scores)
        score_diff[mask, self.left_class] = 1
        score_diff[~mask, :] = 1 / self.num_classes

        return scores + score_diff
