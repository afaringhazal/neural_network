import numpy as np

class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None
    
    def initialize_weights(self):
        if self.initialize_method == "random":
            val = np.random.randn(self.input_size, self.output_size) * 0.01
            print(f'weight(random) :\n {val}')
            return val

        elif self.initialize_method == "xavier":
            # weight = U [-(1/sqrt(n)), 1/sqrt(n)]
            xavier_scale = np.sqrt(2 / (self.input_size + self.output_size))
            # Initialize the weights of the network using Xavier initialization
            val = np.random.normal(scale=xavier_scale, size=(self.input_size, self.output_size))
            print(f'weight(random) :\n {val}')
            return val

        elif self.initialize_method == "he":
            he_scale = np.sqrt(2 / self.output_size)
            # Initialize the weights of the network using He initialization
            val = np.random.normal(scale=he_scale, size=(self.input_size, self.output_size))
            print(f'weight(random) :\n {val}')
            return val
        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        # TODO: Initialize bias with zeros
        val = np.zeros((1, self.output_size))
        print(f'bias :\n {val}')
        return val
    
    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size) L*3
            returns:
                Z: output of the fully connected layer
        """
        # NOTICE: BATCH_SIZE is the first dimension of A_prev
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        # TODO: Implement forward pass for fully connected layer
        # if None: # check if A_prev is output of convolutional layer
        #     batch_size = None
        #     A_prev_tmp = A_prev_tmp.reshape(None, -1).T
        # self.reshaped_shape = A_prev_tmp.shape
        
        # TODO: Forward part
        W, b = self.parameters
        Z = A_prev @ W + b
        return Z
    
    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        # if None: # check if A_prev is output of convolutional layer
        #     batch_size = None
        #     A_prev_tmp = A_prev_tmp.reshape(None, -1).T

        # TODO: backward part
        W, b = self.parameters
        dW = A_prev.T @ dZ / A_prev.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / A_prev.shape[0]
        dA_prev = dZ @ W.T
        grads = [dW, db]
        # reshape dA_prev to the shape of A_prev
        if None:    # check if A_prev is output of convolutional layer
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads
    
    def update(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)


my_class = FC(4,2,"my_FC")
print("successful")