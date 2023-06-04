import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        # Calculate the squared error between y_true and y_pred
        squared_error = np.square(y_true - y_pred)
        # Calculate the mean of the squared error
        mse = np.mean(squared_error)
        return mse
    
    def backward(self, y_pred, y_true):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        # TODO: Implement backward pass for mean squared error loss
        # Calculate the derivative of the squared error between y_true and y_pred
        mse_derivative = 2 * (y_pred - y_true) / len(y_true)
        return mse_derivative