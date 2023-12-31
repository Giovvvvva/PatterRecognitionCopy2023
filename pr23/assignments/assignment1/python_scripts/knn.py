'''
This script contains the skeleton code for the sample assignment-1 for the kNN classifier.
'''
import numpy as np
import matplotlib.pyplot as plt


class KNNClassifier:
    def __init__(self, k = 3):
        """
        k is the number of nearest neighbors to be considered
        """

        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        """
        Train the KNN classifier. For this assignment, you can store the training data itself
        Parameters:
            - X_train : N x D array of training data. Each row is a training sample
            - y_train : N x 1 array of training labels
        Returns: 
            - None
        Save the data in self.X_train and self.y_train
        """
        self.X_train = X_train
        self.y_train = y_train
        # Complete your code here



    def predict(self, X_test):
        """
        Predict labels for test data using the trained classifier
        Parameters:
            - X_test : M x D array of test data. Each row is a test sample
        Returns:
            - predictions : Predicted labels for the test data
        """
        # Complete your code here
        predictions = np.zeros(X_test.shape[0])
        
        for i, x in enumerate(X_test):
            pred_label = self.predict_single_data(x)
            predictions[i] = pred_label
        # Complete your code here

        return np.array(predictions)
    
    def predict_single_data(self, x):
        """
        predict a single data point
        Parameters:
            - x : 1 x D array of test data. Each row is a test sample
        Returns: Predicted label for the test data
        label is the class from y_train that is most represented by the k nearest neighbors
        You can change the return statement to return  the appropriate value

        """
        #distance for single value
        dist = np.linalg.norm(self.X_train - x, axis = 1) #axis was missing here, with courtesy of chgp
        #index according to distances of k neighbors
        index = np.argsort(dist)[:self.k]
        #labels corresponding to the index 
        labels = self.y_train[index]
        #unique labels and their counts
        predict_label = np.bincount(labels).argmax()
        # Complete your code here

        return predict_label

    

    def compute_accuracy(self, X_test, y_test):
        """
        Compute the accuracy of the classifier on the test set
        Parameters:
            - X_test : M x D array of test data. Each row is a test sample
            - y_test : M x 1 array of test labels
        Returns:
            - accuracy : Accuracy of the classifier
        Hint: You should be able to use the predict function get
        the predicted labels for the test data and then compute the accuracy
        """
        prediction = self.predict(X_test)
        count_correct_pred = np.sum(prediction == y_test)
        # Complete your code here

        return count_correct_pred/(len(y_test))

if __name__ == '__main__':
  
    # Test your code here, you can add more test cases to test your implementations. A simple 
    # test case has been provided below:
    # We assume 4 classes and 10 training samples per class with 2 features each and 1 test sample per class
    # We use Guassian random numbers to generate the data with different means for each class
    
    # Generate training data
    np.random.seed(0)

    # Number of classes
    C = 4
    # Number of training samples per class
    N = 101
    means = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])

    # Generate data
    X_train = np.zeros((N*C, 2))
    y_train = np.zeros(N*C).astype(int)

    for i in range(C):
        X_train[i*N:(i+1)*N, :] = np.random.randn(N, 2)/10 + means[i, :]
        y_train[i*N:(i+1)*N] = i


    # Generate test data
    X_test = np.zeros((C, 2))
    y_test = np.zeros(C)

    for i in range(C):
        X_test[i, :] = np.random.randn(1, 2)/10 + means[i, :]
        y_test[i] = i

    # Create a kNN classifier object
    knn = KNNClassifier(k=4)

    # Train the kNN classifier
    knn.train(X_train, y_train)

    # Compute accuracy on the test set
    accuracy = knn.compute_accuracy(X_test, y_test)

    print('Accuracy of the classifier is ', accuracy)

    # Plot the training and the test data
     
    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x')
    plt.show()
    
    # Plot the decision boundary
    # Create a meshgrid of points to be classified
    x_min = -1
    x_max = 2
    y_min = -1
    y_max = 2
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Classify each point of the meshgrid using the trained kNN classifier
    Z = knn.predict(X_grid)

    # Plot the classification boundary
    
    plt.figure()
    plt.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Paired)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x')
    plt.show()
    













