from common import *
from BaseClassifier import BaseClassifier


#Implimenting Linear Discriminant Analysis classifier from scratch

class LinearDiscriminantAnalysisClassifier(BaseClassifier):

  def __init__(self):

    self.weight = None
    self.bias = None
    self.threshold = None

  def train(self, X_train: np.ndarray, y_train: np.ndarray):

    X_loan_approved = X_train[y_train == 1]         # This hold the feature values wheren ApprovedLoan is value 1
    X_loan_not_approved = X_train[y_train == 0]     # This hold the feature values where ApprovedLoan is value 0

    mean_loan_approved = np.mean(X_loan_approved, axis = 0)        #Calculating the mean of the loan dataset which will we use to find the covarience of this metrices. 
    mean_loan_not_approved = np.mean(X_loan_not_approved, axis = 0)

    cov_loan_approved = np.cov(X_loan_approved.T)               # Calculating the covarience matrix of Loan approved features
    cov_loan_not_approved = np.cov(X_loan_not_approved.T)       #Calculating the covarience matrix of Loan not approved features

    within_class_scatter_matrix = cov_loan_approved + cov_loan_not_approved #claculating the within class scatter matrix

    self.weight = np.linalg.inv(within_class_scatter_matrix) @ (mean_loan_approved - mean_loan_not_approved) #The weight vector for the linear discriminant is calculated here. It is obtained by multiplying the inverse of the within-class scatter matrix by the difference between the mean vectors of the two classes. This weight vector will be used to project the data onto a lower-dimensional space.

    self.bias = 0.5 * (mean_loan_approved + mean_loan_not_approved) @ self.weight  #we compute the bias here 

    projection_of_train_data = X_train @ self.weight + self.bias  #  Here the training data is projected into the new feature space defined by the weight vector.

    projection_approved = projection_of_train_data[y_train == 1]  # This hold the projected data for samples with loan approved
    projection_not_approved = projection_of_train_data[y_train == 0] # This hold the projected data for samples with loan not approved

    self.threshold = (np.mean(projection_approved) + np.mean(projection_not_approved))/2  # Calculating threshold for making classification decisions  as the average of the mean projections for both classes.

    logger.info(f"Training LDA is successfully completed")
    logger.info(f'weight: {self.weight}')
    logger.info(f'bias: {self.bias}')
    logger.info(f'threshold: {self.threshold}')

  def project(self, X: np.ndarray):      # This method give the projection of each sample onto the direction give by the weight vector.

    return X @ self.weight + self.bias


  def get_model_parameters(self):  

    return self.weight, self.bias

  def error_rate_calculation(self, X_test: np.ndarray, y_test: np.ndarray, thresholds: np.ndarray):  # This is the error function which we use to print and plot the type 1 and type 2 error.

    projections = self.project(X_test) 
    type_1_error_rate = []
    type_2_error_rate = []
    threshold_values = []

    for threshold in thresholds:
      predictions = np.where(projections >= threshold, 1, 0)
      threshold_values.append(threshold)
      type_1_error = np.sum((predictions == 1) & (y_test == 0))/np.sum(y_test == 0)   # Here the type 1 error which is predicted  1 or approved but it was actually not approved. 
      type_2_error = np.sum((predictions == 0) & (y_test == 1))/np.sum(y_test == 1)   # Here the type 2 error which is prediceted 0 or not approved but it was actually approved.

      type_1_error_rate.append(type_1_error)
      type_2_error_rate.append(type_2_error)

    return np.array(type_1_error_rate), np.array(type_2_error_rate)

  def plot_error_rate(self, X_test: np.ndarray, y_test: np.ndarray): 
    projections = self.project(X_test)
    
    min_threshold_value = np.min(projections)
    max_threshold_value = np.max(projections)

    thresholds = np.linspace(min_threshold_value, max_threshold_value, 100)

    type_1_error_rate, type_2_error_rate = self.error_rate_calculation(X_test, y_test, thresholds)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, type_1_error_rate, label='Type I Error Rate (False Positive Rate)', color='blue')
    plt.plot(thresholds, type_2_error_rate, label='Type II Error Rate (False Negative Rate)', color='red')
    plt.title('Type I and Type II Error Rates vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid()
    plt.show()


  def predict(self, X_test: np.ndarray, custom_threshold):           #Prediction method to get the predictions of test_data
    projection_of_test_data = self.project(X_test)
    predictions = np.where(projection_of_test_data >= custom_threshold, 1, 0) # We predict our sample beased on the threshold value
    return predictions

  def find_best_threshold(self, X_test: np.ndarray, y_test: np.ndarray, thresholds: np.ndarray):
    best_threshold = None
    best_f1_score = 0

    for threshold in thresholds:
      predictions = np.where(self.project(X_test) >= threshold, 1, 0)       # Here we generate the predictions by projecting the test data on weight vector
      current_f1_score = f1_score(y_test, predictions)          # We calculate the f1 score between the test labels and predicted labels 

      if current_f1_score > best_f1_score:                        # we check if the current f1 score is better than previous one 
        best_f1_score = current_f1_score
        best_threshold = threshold                                # we return threshould which contribute to the highest f1 score. 

    return best_threshold, best_f1_score
