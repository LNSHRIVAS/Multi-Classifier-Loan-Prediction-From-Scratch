from common import *

# Support Vector Machine (SVM) from scratch implementation

class SupportVectorMachineClassifier():
  def __init__(self, learning_rate = 0.001, num_iterations = 1000, C = 0.01):

    #This value of C which is 0.01 is what I have found by doing experiment and manually changing C value and finally it turned out that c = 0.01 resutls in the maximum accuracy 

    # Initialize the hyperparameters
    # C: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error
    # learning_rate: Controls how much to adjust the weights with each iteration
    # num_iterations: Number of passes over the entire training dataset
    self.C = C
    self.learning_rate = learning_rate
    self.num_iterations = num_iterations
    self.weights = None  # Weight vector initialized later
    self.bias = None     # Bias term initialized later

  def fit(self, x_train, y_train):
    # Fit the model to the training data
    num_samples, num_features = x_train.shape  # Get number of samples and features in the training set
    y_label = np.where(y_train <= 0, -1, 1)    # Convert labels to -1 and 1 for SVM
    self.weights = np.zeros(num_features)      # Initialize weights to zero
    self.bias = 0                              # Initialize bias to zero

    # Objective function: Minimize ||w||^2 + C * sum(max(0, 1 - y*(w*x - b)))
    # where ||w||^2 is the regularization term (encouraging a larger margin)
    # and C * sum(...) penalizes misclassifications.
    
    # Perform gradient descent for a fixed number of iterations
    for _ in range(self.num_iterations):
      for i in range(num_samples):
        # Check if the sample satisfies the SVM condition (inside the margin or correctly classified)
        if (y_label[i] * (np.dot(x_train[i], self.weights) - self.bias)) >= 1:
          # If the point is correctly classified with margin, update weights with regularization term only
          # This represents minimizing the ||w||^2 part of the objective function
          self.weights -= self.learning_rate * (2 * self.C * self.weights)
        else:
          # If misclassified, adjust weights and bias to penalize misclassification
          # This represents minimizing both ||w||^2 and the hinge loss component, C * max(0, 1 - y*(w*x - b))
          self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(x_train[i], y_label[i]))
          self.bias -= self.learning_rate * y_label[i]

  def predict(self, x_test):
    # Predict the class for each sample in x_test
    output = np.dot(x_test, self.weights) - self.bias  # Linear decision function
    predictions = np.where(output >= 0, 1, -1)          # Classify as 1 or -1 based on sign of output
    return np.where(predictions == -1, 0, 1)            # Convert -1 back to 0 for consistency with original labels

  def get_accuracy_and_predictions(self, x_test, y_test):
    # Calculate accuracy and predictions for the test set
    predictions = self.predict(x_test)
    accuracy = np.mean(predictions == y_test)  # Calculate the percentage of correct predictions
    return accuracy, predictions
