import numpy as np
from abc import ABC, abstractmethod

from abs_layer import Abs_layer


class Activation(Abs_layer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Forward pass for activation function.
            args:
                Z: input to the activation function
            returns:
                A: output of the activation function
        """
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        pass


class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
            args:
            x: input to the activation function
            returns:
                sigmoid(x)
        """
        sig = 1/(1 + np.exp(-Z))
        return sig

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for sigmoid activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        A = self.forward(Z)
        dZ = A * (1-A) * dA
        return dZ

class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
            args:
                x: input to the activation function
            returns:
                relu(x)
        """
        # TODO: Implement ReLU activation function
        return np.where(Z > 0, Z, 0)

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        # TODO: Implement backward pass for ReLU activation function
        dZ = dA
        dZ[Z <= 0] = 0

        return dZ
    
    

class Tanh(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.
            args:
                x: input to the activation function
            returns:
                tanh(x)
        """
        # TODO: Implement tanh activation function
        exp_z = np.exp(Z)
        negative_exp_z = np.exp(-Z)
        A = (exp_z - negative_exp_z)/(exp_z + negative_exp_z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for tanh activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        A = self.forward(Z)

        # TODO: Implement backward pass for tanh activation function
        dZ = (1 - (np.square(A))) * dA
        return dZ
    
class LinearActivation(Activation):
    def linear(Z: np.ndarray) -> np.ndarray:
        """
        Linear activation function.
            args:
                x: input to the activation function
            returns:
                x
        """
        # TODO: Implement linear activation function
        A = Z
        return A

    def backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray: #TODO Fixe me
        """
        Backward pass for linear activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        # TODO: Implement backward pass for linear activation function
        dZ = dA
        return dZ

def get_activation(activation: str) -> tuple:
    """
    Returns the activation function and its derivative.
        args:
            activation: activation function name
        returns:
            activation function and its derivative
    """
    if activation == 'sigmoid':
        return Sigmoid
    elif activation == 'relu':
        return ReLU
    elif activation == 'tanh':
        return Tanh
    elif activation == 'linear':
        return LinearActivation
    else:
        raise ValueError('Activation function not supported')