from common import *

#Not so important class

class BaseClassifier:
    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError