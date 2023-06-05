from abs_layer import Abs_layer
from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC
from typing import Dict, Set, List
from activations import Activation, get_activation
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model: Dict[str, Abs_layer] = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)
    
    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        # TODO: Implement check if the layer is a layer

        return isinstance(layer, FC) or isinstance(layer,Conv2D) or isinstance(layer,MaxPool2D)

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        # TODO: Implement check if the layer is an activation
        return isinstance(layer,Activation)
    
    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x
        # TODO: Implement forward pass through the model
        # NOTICE: we have a pattern of layers and activations
        # tmp.append(x)
        # values = self.model.values()
        keys = self.layers_names
        for l in range(0, len(keys), 2):
            Z = self.model[keys[l]].forward(A)
            tmp.append(Z)    # hint add a copy of Z to tmp
            A = self.model[keys[l+1]].forward(Z)
            tmp.append(A)    # hint add a copy of A to tmp
        return tmp
    
    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        # TODO: Implement backward pass through the model
        # NOTICE: we have a pattern of layers and activations
        # for from the end to the beginning of the tmp list
        names = self.layers_names
        i = len(names)
        for l in range(len(tmp), 0, -2):
            if l > 2:
                Z, A_prev = tmp[l - 2], tmp[l - 3]
            else:
                Z, A_prev = tmp[l - 2], x
            dZ = self.model[names[i-1]].backward(dA, Z)  # call backward activation function
            dA, grad = self.model[names[i-2]].backward(dZ, A_prev)  #
            grads[self.layers_names[i - 2]] = grad
            i -= 2
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        keys = self.model.keys()
        for name in keys:
            if self.is_layer(self.model[name]):
                if not isinstance(self.model[name], MaxPool2D):    # hint check if the layer is a layer and also is not a maxpooling layer
                    self.model[name].update(self.optimizer, grads)
    
    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        # TODO: Implement one epoch of training
        tmp = self.forward(x)
        size_tmp = len(tmp)
        AL = tmp[size_tmp - 1]
        # print(f'AL : {AL}')
        loss = self.criterion.forward(AL, y)  # calculate cost y_pred and y_true
        # now do backward
        dAL = self.criterion.backward(AL, y)  # مشتق cost نسبت به AL
        grads = self.backward(dAL, tmp, x)
        # print(f'grads : {grads}')
        # Update
        self.update(grads)
        return loss

    
    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)
        
    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        # TODO: Implement batch
        last_index = None   # hint last index of the batch check for the last batch
        batch_order = order[index * batch_size: (index+1) * batch_size]
        # NOTICE: inputs are 4 dimensional or 2 demensional
        if len(X.shape) == 4:
            bx = X[batch_order, :, :, :]
            by = y[batch_order]
            return bx, by
        else:
            bx = X[batch_order, :]
            by = y[batch_order]
            return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        # TODO: Implement compute loss
        m = X.shape[0]  # give the number of rows in X
        order = self.shuffle(m, True)
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b, order)
            tmp = self.forward(bx)
            size_tmp = len(tmp)
            AL = tmp[size_tmp-1]
            cost += self.criterion.forward(AL, by)
        return cost / (m // batch_size)  # normalize

    def train(self, X, y, epochs, val=None, batch_size= 5 , shuffling=False, verbose=1, save_after=None): # batch size nabayad bishtar bashe?
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        # TODO: Implement training
        train_cost = []
        val_cost = []
        # NOTICE: if your inputs are 4 dimensional m = X.shape[0] else m = X.shape[1]
        if len(X.shape) == 4:
            m = X.shape[0] # TODO
        else:
            m = X.shape[0]  # TODO
        for e in range(1, epochs + 1):
            order = self.shuffle(m, shuffling)
            cost = 0
            loss =0
            for b in range(m // batch_size):
                bx, by = self.batch(X, y, batch_size, b, order)
                loss = self.one_epoch(bx, by)
                cost += loss
            cost = cost / (m // batch_size)
            print(f'cost  : {cost}')
            train_cost.append(cost)
            if val is not None:
                val_cost.append(None)
            if verbose != False:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost
    
    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        # TODO: Implement prediction
        # return None
        tmp = self.forward(X)
        siz = len(tmp)
        return tmp[siz-1]



