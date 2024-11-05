from common import *
class DataPreprocessing:

  def __init__(self, data: pd.DataFrame):
    self.data = data

  def handle_missing_values(self):
    self.data.fillna(self.data.mean(), inplace = True)

  def standardize_data(self):
    mean = self.data.mean()
    std = self.data.std()
    standardized_data = (self.data - mean) / std
    standardized_data = standardized_data.iloc[:,:].values
    logger.info("Standardized column: {column} (mean: {mean}, std: {std})")
    return standardized_data

  def normalize_data(self):
    min_value = self.data.min()
    max_value = self.data.max()
    normalized_data = (self.data - min_value) / (max_value - min_value)
    normalized_data = normalized_data.iloc[:,:].values
    logger.info("Normalized column: {column} (min: {min_value}, max: {max_value})")
    return normalized_data

  def transform_data(self):
    self.handle_missing_values()
    standardized_data = self.standardize_data()
    normalized_data = self.normalize_data()
    return standardized_data, normalized_data