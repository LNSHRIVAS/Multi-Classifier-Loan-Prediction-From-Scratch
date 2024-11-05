from common import *

#This is an additional class which I created solely because I wanted to practice the unittest library this just checks the data loading and verify it by doing a small testing on it
#Ensuring that every function in data loader and data preprocessing is working fine that is it.
class TestDataLoader(unittest.TestCase):

  def setUp(self):

    self.train_data_path = '/content/drive/MyDrive/LoanData/mock_train.csv'
    self.test_data_path = '/content/drive/MyDrive/LoanData/mock_test.csv'
    self.label_column_name = 'LoanApproved'
    self.training_data_read = pd.read_csv(training_data) 
    self.testing_data_read = pd.read_csv(testing_data)
    self.temp_train_data = self.training_data_read.sample(n=100)
    self.temp_test_data = self.testing_data_read.sample(n=100)

    self.temp_train_data.to_csv(self.train_data_path, index=False)
    self.temp_test_data.to_csv(self.test_data_path, index=False)

    self.data_loader = DataLoader(self.train_data_path, self.test_data_path, self.label_column_name)
    self.data_loader.verify_data_path()

  def tearDown(self):
    if os.path.exists(self.train_data_path):
      os.remove(self.train_data_path)
    if os.path.exists(self.test_data_path):
      os.remove(self.test_data_path)

  def testing_verify_data_path(self):
    try:
      self.data_loader.verify_data_path()
      logger.info("Data path verification successful.")
    except Exception as e:
      logger.error(f"Data path verification failed: {e}")

  def testing_load_training_data(self):
    train_data_features, train_data_labels = self.data_loader.load_training_data()
    pd.testing.assert_frame_equal(train_data_features.reset_index(drop=True), self.temp_train_data.drop(self.label_column_name, axis=1).reset_index(drop=True), check_dtype=True)
    pd.testing.assert_series_equal(train_data_labels.reset_index(drop=True), self.temp_train_data[self.label_column_name].reset_index(drop=True), check_dtype=True)
    logger.info("Training data loaded successfully.")

  def testing_load_testing_data(self):
    test_data_features, test_data_labels = self.data_loader.load_testing_data() 
    pd.testing.assert_frame_equal(test_data_features.reset_index(drop=True), self.temp_test_data.drop(self.label_column_name, axis=1).reset_index(drop=True), check_dtype=True)
    pd.testing.assert_series_equal(test_data_labels.reset_index(drop=True), self.temp_test_data[self.label_column_name].reset_index(drop=True), check_dtype=True)
    logger.info("Testing data loaded successfully.")

