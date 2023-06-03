import numpy as np
from losses.binarycrossentropy import BinaryCrossEntropy

def compute_binary_cross_entropy_test():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
    error_class = BinaryCrossEntropy()
    bce = error_class.compute(y_pred, y_true)
    print("Binary cross-entropy loss:", bce)

def backward_binary_cross_entropy_test():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7])

    error_class = BinaryCrossEntropy()
    mse_deriv = error_class.backward(y_pred, y_true)
    print("Derivative of minimum squared error:", mse_deriv)


compute_binary_cross_entropy_test()
backward_binary_cross_entropy_test()