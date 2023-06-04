import numpy as np
from losses.meansquarederror import MeanSquaredError

def compute_min_squared_error_test():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    error_class = MeanSquaredError()

    mse = error_class.compute(y_pred, y_true)
    print("Minimum squared error:", mse)

def backward_min_squared_error_test():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    error_class = MeanSquaredError()
    mse_deriv = error_class.backward(y_pred, y_true)
    print("Derivative of minimum squared error:", mse_deriv)



compute_min_squared_error_test()
backward_min_squared_error_test()