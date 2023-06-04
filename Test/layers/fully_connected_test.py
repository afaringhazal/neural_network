import numpy as np

from layers.fullyconnected import FC


def initial_weight_random():
    my_fully_connected_class = FC(4, 3, "l1", "random")
    print(my_fully_connected_class.parameters)


def initial_weight_xavier():
    my_fully_connected_class = FC(4, 3, "l1", "xavier")
    print(my_fully_connected_class.parameters)


def initial_weight_he():
    my_fully_connected_class = FC(4, 3, "l1", "he")
    print(my_fully_connected_class.parameters)


def forward_test():
    my_fully_connected_class = FC(4, 3, "l1")
    A_prev = np.array([1, 2, 4, 5])
    val = my_fully_connected_class.forward(A_prev)
    print(val)


def backward_test():
    my_fully_connected_class = FC(4, 3, "l1")
    weight , bias = my_fully_connected_class.parameters
    dZ = np.array([[1, 3,  6]])
    A_prev = np.array([[1, 2, 4, 5]])
    dA_prev, grades = my_fully_connected_class.backward(dZ, A_prev)
    print(f'calculate dA_prev by method : \n {dA_prev}')
    print(f'calculate grades by FC :\n {grades}')
    # now calculate us
    print("-----------------------")
    print(f'w is :\n {weight}')
    print(f'bais is :\n {bias}')


initial_weight_random()
initial_weight_xavier()
initial_weight_he()
forward_test()
backward_test()

