from common import *

class DecisionTreeClassifier():
  def __init__ (self, max_depth, feature_names):
    self.optimal_gini = float('inf')   # For now we are setting the gini impurtity to highest to compare it with our latest values and we will update it if our new gini value turns less as our goal is to minize the gini impurity.
    self.optimal_feature_index = None  
    self.optimal_split_threshold = None
    self.max_depth = max_depth
    self.decision_tree = None
    self.minmum_samples_split = 2
    self.minmum_samples_leaf = 1
    self.feature_names = feature_names
  
  def gini_impurity(self, y):   #Calculating the gini impurity as disscussed in the class using those formulas. 
    class_labels, label_counts = np.unique(y, return_counts = True)  
    class_probabilities = label_counts / label_counts.sum()
    gini_impurity = 1 - np.sum(class_probabilities ** 2)
    return gini_impurity

  def information_gain(self, y, left_indices, right_indices): #Calulating the information gain as discussed in the classroom. 
    parent_gini_impurity = self.gini_impurity(y)
    left_gini_impurity = self.gini_impurity(y[left_indices])
    right_gini_impurity = self.gini_impurity(y[right_indices])
    weighted_gini_impurity = (len(left_indices) / len(y)) * left_gini_impurity + (len(right_indices) / len(y)) * right_gini_impurity
    information_gain = parent_gini_impurity - weighted_gini_impurity
    return information_gain

  def optimal_split(self, x, y):
    # Determine the best feature and threshold to split on, maximizing information gain
    best_information_gain = -1  # Initialize the best information gain to a very low value
    best_feature_index = None  # Store the best feature index
    best_threshold = None  # Store the best threshold value
    sample, feature = x.shape  # Get the number of samples and features

    # Iterate over each feature to find the best split
    for feature_index in range(feature):
      feature_values = x[:, feature_index]
      unique_values = np.unique(feature_values)  # Unique values for thresholding

      # Iterate over each unique value to test it as a threshold
      for threshold in unique_values:
        left_indices = np.where(x[:, feature_index] < threshold)[0]
        right_indices = np.where(x[:, feature_index] >= threshold)[0]

        # Skip if a split does not create both left and right subsets
        if len(left_indices) == 0 or len(right_indices) == 0:
          continue

        # Calculate the information gain for this split
        information_gain = self.information_gain(y, left_indices, right_indices)

        # Update the best split if this split has a higher information gain
        if information_gain > best_information_gain:
          best_information_gain = information_gain
          best_feature_index = feature_index
          best_threshold = threshold

    return best_feature_index, best_threshold

  def generate_decision_tree(self, x, y, depth=0):
      # Recursively build the decision tree
      # If all samples have the same label, return a leaf node with that label
      if len(np.unique(y)) == 1:
        return {'class': y[0]}

      # Stop if the maximum depth is reached or if splitting further is not meaningful
      if self.max_depth is not None and depth >= self.max_depth:
        return {'class': np.argmax(np.bincount(y))}

      if len(y) < self.minmum_samples_split:
        return {'class': np.argmax(np.bincount(y))}

      # We Determine the best feature and threshold to split on
      optimal_index, optimal_threshold = self.optimal_split(x, y)

      # If no valid split is found, return a leaf node with the majority class
      if optimal_index is None:
        return {'class': np.argmax(np.bincount(y))}

      # Spliting the data into left and right subsets
      left_indices = np.where(x[:, optimal_index] < optimal_threshold)[0]
      right_indices = np.where(x[:, optimal_index] >= optimal_threshold)[0]

      # Checking for minimum samples at leaf nodes
      if len(left_indices) < self.minmum_samples_leaf or len(right_indices) < self.minmum_samples_leaf:
        return {'class': np.argmax(np.bincount(y))}

      # Storing the question for the current node and this will be the ouput questions which was asked in the question by professor.
      question = f"Is {optimal_index} <= {optimal_threshold}"

      # Recursively building the left and right branches of the tree
      left_node_subtree = self.generate_decision_tree(x[left_indices], y[left_indices], depth + 1)
      right_node_subtree = self.generate_decision_tree(x[right_indices], y[right_indices], depth + 1)

      # Returning the structure of this node in the tree
      return { 'question': question, 'feature_index': optimal_index, 'threshold': optimal_threshold, 'left': left_node_subtree, 'right': right_node_subtree}
      

  def fit(self, x, y):
      self.decision_tree = self.generate_decision_tree(x, y)

  def predict_sample(self, x, tree):
      if 'class' in tree:
        return tree['class']
      # Now we Traverse the left or right subtree based on the feature threshold
      if x[tree['feature_index']] < tree['threshold']:
        return self.predict_sample(x, tree['left'])

      else:
        return self.predict_sample(x, tree['right'])

  def predict(self, x): # This is our predict method we predict the test samples.
      predictions = [self.predict_sample(sample, self.decision_tree) for sample in x]
      return np.array(predictions)

  def print_decision_tree(self, tree = None, depth = 0):
     if tree is None:
            tree = self.decision_tree

     if 'class' in tree:
            print(f"{' ' * depth * 2}Predict: {tree['class']}")
     else:
            question = f"Is {self.feature_names[tree['feature_index']]} <= {tree['threshold']}" 
            print(f"{' ' * depth * 2}{question}")
            self.print_decision_tree(tree['left'], depth + 1)
            self.print_decision_tree(tree['right'], depth + 1)

  def get_accuracy_and_predictions(self, x_test, y_test):
    # Calculate accuracy and predictions for the test set
    predictions = self.predict(x_test)
    accuracy = np.mean(predictions == y_test)  # Calculate the percentage of correct predictions
    return accuracy, predictions