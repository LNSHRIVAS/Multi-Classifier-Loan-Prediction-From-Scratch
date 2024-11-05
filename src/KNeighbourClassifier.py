from common import *

class KNeighbourClassifier():
  def __init__(self, k):
    self.k = k

  def fit(self, x_train, y_train):
    
        self.x_train = x_train
        self.y_train = y_train

  def calculate_euclidean_distance(self, x1, x2):  # we are using euclidena distance to find the distance between the data potins. 
    return np.sqrt(np.sum((x1 - x2) ** 2))

  def predict(self, x_test):

    predcitions_for_test_data = []

    for test_data in x_test:

      distances = [self.calculate_euclidean_distance(test_data, train_data) for train_data in self.x_train]  # We compute the distance of each test sample to the each training sample.

      indices_of_k_nearest_neighbours = np.argsort(distances)[:self.k]    # Now we sort this distances and find the k nearest neighbours. 
      K_Nearest_Neighbors = self.y_train[indices_of_k_nearest_neighbours] # Now we Retrieves the labels of the k nearest neighbors from y_train

      unique_labels, label_counts = np.unique(K_Nearest_Neighbors, return_counts = True) #using np.unique we count the occurrences of each label among the neighbors
      predictions = unique_labels[np.argmax(label_counts)] #The label with the highest count is chosen as the prediction for the test sample

      predcitions_for_test_data.append(predictions)

    return predcitions_for_test_data

  def get_accuracy_and_predictions(self, x_test, y_test):  #This is a common function in every classifier to get accuracy and predictions.
    predictions = np.array(self.predict(x_test))
    accuracy = np.mean(predictions == y_test)
    return accuracy, predictions