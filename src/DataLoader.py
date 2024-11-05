import pandas as pd
from common import *

# This class is simple data loading class which I create to automate the processes of retirving data and converting it into feature and labels.

class DataLoader:

  def __init__(self, train_data_path: str ,test_data_path: str, label_column_name: str ):
    self.train_data_path = train_data_path
    self.test_data_path = test_data_path
    self.training_data = None
    self.testing_data = None
    self.training_data_features = None
    self.testing_data_features = None
    self.training_data_labels = None
    self.testing_data_labels = None
    self.target_label_column_name = label_column_name
    self.feature_names = None
    self.target_column = label_column_name

  def verify_data_path(self):  # This is to verify if the data is present at the listed path ?
      for data_path in [self.train_data_path, self.test_data_path]:
        if not os.path.exists(data_path):
          logger.error(f"File error: {data_path}")
          raise FileNotFoundError(f"File not available: {data_path}")
        else:
          logger.info(f"File is available: {data_path}")

  def load_training_data(self) -> pd.DataFrame:
    try:

      logger.info(f"Loading training data...")
      self.training_data = pd.read_csv(self.train_data_path)
      self.training_data_features = self.training_data.drop(self.target_label_column_name, axis = 1)
      self.training_data_labels = self.training_data[self.target_label_column_name]
      self.feature_names = self.training_data.columns.drop(self.target_column).tolist()
      logger.info(f"Training data loaded successfully.")
      return self.training_data_features, self.training_data_labels.values, self.feature_names

    except Exception as e:
      logger.error(f"Error loading training data: {e}")
      raise

  def load_testing_data(self) -> pd.DataFrame:
      try:
        logger.info("Loading testing data...")
        self.testing_data = pd.read_csv(self.test_data_path)
        self.testing_data_features = self.testing_data.drop(self.target_label_column_name, axis = 1)
        self.testing_data_labels = self.testing_data[self.target_label_column_name]
        logger.info(f"Testing data loaded successfully.")
        return self.testing_data_features, self.testing_data_labels.values

      except Exception as e:
        logger.error(f"Error loading testing data: {e}")

