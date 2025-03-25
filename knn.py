
import numpy as np
import faiss
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from GreedyBoosting import GreedyBoosting

train_data = pd.read_csv('train.csv').values
test_data = pd.read_csv('test.csv').values
validation_data = pd.read_csv('validation.csv').values
ad_test_data = pd.read_csv('AD_test.csv').values


class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.index = None
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = []

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        # pass
        self.index.add(self.X_train)

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        # Using the faiss search algorythm to determine the distances between a set of points and its k nearest neighbors
        # It returns the distances in ascending order and its indices for better manipulation
        distance, indices = self.index.search(X, self.k)
        # print(indices)
        return distance, indices

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        #### YOUR CODE GOES HERE ####
        distance, indices = self.knn_distance(X)

        predictions_array = []

        # We go over all the points, and we determine the probability of a point to be a certain label by summing over all the neighbors
        for indice in indices :
            label_count = np.bincount(self.Y_train[indice]).argmax()
            predictions_array.append(label_count)

        return np.array(predictions_array)
        

def display_knn_plots(train_features, train_labels, test_features, test_labels):

    from helpers import plot_decision_boundaries

    # l2_kmax = predictions_accuracy[1].index(np.max(predictions_accuracy[1]))
    # l2_kmin = predictions_accuracy[1].index(np.min(predictions_accuracy[1]))
    # l1_kmax = predictions_accuracy[0].index(np.max(predictions_accuracy[0]))

    # I did the numbers k manually since it's becoming a little bit complex to do them automatically ( watch two lines earlier)
    plots = [
        {"distance_metric": "l2", "k": 1, "title": "L2, k=1 (kmax)"},  # (i)
        {"distance_metric": "l2", "k": 3000, "title": "L2, k=3000 (kmin)"},  # (ii)
        {"distance_metric": "l1", "k": 1, "title": "L1, k=1 (kmax)"}  # (iii)
    ]
    
    for plot in plots:
        knn = KNNClassifier(plot["k"], plot["distance_metric"])
        knn.fit(train_features, train_labels)
        plot_decision_boundaries(knn, test_features, test_labels, plot["title"])


def knn_classification():

    # train_data_numpy, train_col_names = helpers.read_data_demo('train.csv')
    # test_data_numpy, test_col_names = helpers.read_data_demo('test.csv')

    train_features = train_data[:, :2] 
    train_labels = train_data[:, 2].astype(int)  
    test_features = test_data[:, :2]
    test_labels = test_data[:, 2].astype(int)

    result_table = []

    for distance_method in ["l1",'l2']:
        result_row = [] # Should be only 2
        for k in [1,10,100,1000,3000]:

            classifier = KNNClassifier(k, distance_method)
            classifier.fit(train_features, train_labels)

            predictions = classifier.predict(test_features)
            # predictions == label will compare between each values of the lists and will return a list of boolean
            # Then np.mean does the average of the booleans ex : [1,0,1,1] -> 1+0+1+1 / 4 -> 0.75 and that's the accuracy
            accuracy = np.mean(predictions == test_labels)

            result_row.append(accuracy)

        result_table.append(result_row)
    
    results_table = pd.DataFrame(result_table, columns=[1,10,100,1000,3000], index=["l1",'l2'])
    results_table.index.name = 'distance_metric'
    results_table.columns.name = 'k'

    print(results_table)

    display_knn_plots(train_features, train_labels, test_features, test_labels)

def knn_anomalies_identification():

    train_features = train_data[:, :2] 
    train_labels = train_data[:, 2].astype(int)  
    test_features = ad_test_data[:, :2]

    knn_classifier = KNNClassifier(5, "l2")
    knn_classifier.fit(train_features, train_labels)

    distance, indices = knn_classifier.knn_distance(test_features)

    anomaly_scores = np.sum(distance, axis=1)
    best_50_anomalies = np.argsort(anomaly_scores)[-50:]

    # Creating an anomaly codex
    anomalies = np.zeros(test_features.shape[0], dtype=bool)
    anomalies[best_50_anomalies] = True

    # Normal points in blue, anomalies in red
    normal_points = test_features[~anomalies]
    anomalous_points = test_features[anomalies]

    # Plotting the anomalies and normal points
    plt.figure(figsize=(10, 8))

    # Plot normal points (blue)
    plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='Normal', alpha=0.7)

    # Plot anomalies (red)
    plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], c='red', label='Anomaly', alpha=0.7)

    # Plot train points (black with low opacity)
    plt.scatter(train_features[:, 0], train_features[:, 1], c='black', alpha=0.01, label='Train Data')

    # Adding labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Anomaly Detection: Normal vs Anomalous Points')
    plt.legend()

    # Display the plot
    plt.show()


def display_plot_accuracy(model, train_features, train_labels, test_features, trest_labels, max_stumps):
    """
    Present a graph of the accuracy as a function of the number of decision
    stumps, showing results for both the training set and the test set on the
    same plot.
    Args:
        model:
        train_features:
        train_labels:
        test_features:
        trest_labels:
        max_stumps:

    Returns:

    """
    train_accuracies = []
    test_accuracies = []

    for num in range(1, max_stumps + 1):
        train_predictions = model.predict_with_stumps(train_features, num)
        test_predictions = model.predict_with_stumps(test_features, num)

        train_accuracies.append(np.mean(train_predictions == train_labels))
        test_accuracies.append(np.mean(test_predictions == trest_labels))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_stumps + 1), train_accuracies, label="Training Accuracy", marker='o')
    plt.plot(range(1, max_stumps + 1), test_accuracies, label="Testing Accuracy", marker='s')
    plt.xlabel("Decision Stumps")
    plt.ylabel("Accuracy")
    plt.title("Training set and test set of Decision Stumps")
    plt.legend()
    plt.grid(True)
    plt.show()


def display_data_class_predictions(model, train_features):
    """
    Scatter plot of training data, colored by predicted classes for different numbers of stumps.

    Parameters:
    - model: Trained GreedyBoosting model
    - X_train, y_train: Training data and labels
    - iterations: List of numbers of stumps to visualize
    """
    for num in [1, 5, 15, 25]:
        predictions = model.predict_with_stumps(train_features, num)

        plt.figure(figsize=(10, 6))
        for cls in np.unique(predictions):
            mask = predictions == cls
            plt.scatter(train_features[mask, 0], train_features[mask, 1], label=f"Class {cls}", alpha=0.7)

        plt.title(f"Training Data - Predicted Classes (Iteration {num})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.show()


def greedy_boosting():

    # Split the dataset into training and testing sets (90% train, 10% test)
    train_features, test_features, train_labels, test_labels = (
        train_test_split(train_data[:, :2], train_data[:, 2].astype(int) , test_size=0.1, random_state=42))
    
    # Initialize and train the greedy boosting model
    number_of_classes = len(np.unique(train_labels))

    greedy_boosting_model = GreedyBoosting(num_classes=number_of_classes, num_stumps=25, num_thresholds=20)
    greedy_boosting_model.fit(train_features, train_labels)
    
    # Evaluate the model
    train_predictions = greedy_boosting_model.predict(train_features)
    test_predictions = greedy_boosting_model.predict(test_features)
    
    train_accuracy = np.mean(train_predictions == train_labels)
    test_accuracy = np.mean(test_predictions == test_labels)
    
    print(f"Final Training Accuracy: {train_accuracy}")
    print(f"Final Testing Accuracy: {test_accuracy}")

    # Call the plotting functions
    display_plot_accuracy(greedy_boosting_model, train_features, train_labels, test_features, test_labels, max_stumps=25)
    display_data_class_predictions(greedy_boosting_model, train_features)


def main():
    np.random.seed(0)

    # knn_classification()
    # knn_anomalies_identification()
    greedy_boosting()



if __name__ == "__main__":
    main()