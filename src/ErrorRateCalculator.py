from common import * 

class ErrorRateCalculator:

  def errore_rate_calculation(self,y_test, predictions):
    type_1_error_rate = np.sum((predictions == 1) & (y_test == 0))/np.sum(y_test == 0)
    type_2_error_rate = np.sum((predictions == 0) & (y_test == 1))/np.sum(y_test == 1)
    return type_1_error_rate, type_2_error_rate