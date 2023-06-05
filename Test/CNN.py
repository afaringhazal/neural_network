import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from activations import Tanh, Sigmoid
from layers.convolution2d import Conv2D
from layers.fullyconnected import FC
from losses.binarycrossentropy import BinaryCrossEntropy
from model import Model
from optimizers.gradientdescent import GD

print(os.listdir("../datasets/MNIST/2/"))

list_of_2_images_file_names = os.listdir("../datasets/MNIST/2/")[:20]
list_of_5_images_file_names = os.listdir("../datasets/MNIST/5/")[:20]
fives = np.zeros(shape=(len(list_of_5_images_file_names), 28, 28, 1))
twos = np.zeros(shape=(len(list_of_2_images_file_names), 28, 28, 1))
for i in range(len(list_of_2_images_file_names)):
    image = plt.imread(f"../datasets/MNIST/2/{list_of_2_images_file_names[i]}")
    twos[i, :, :, 0] = image
for i in range(len(list_of_5_images_file_names)):
    fives[i, :, :, 0] = plt.imread(f"../datasets/MNIST/5/{list_of_5_images_file_names[i]}")
images = np.concatenate((twos, fives), axis=0)
target = [([1] * len(twos)) + ([0] * len(fives))]
target = np.array(target).reshape(len(target[0]), 1)
print(target)

# target = target.reshape(target.shape[0], 1, 1, 1)
print(target.shape)
#
# im: np.ndarray = plt.imread("../datasets/MNIST/2/img_22.jpg")
# im: np.ndarray = im.reshape(1, im.shape[0], im[, 1)
# im = im.astype('float64')

# print(type(im))
# plt.imshow(im)
# plt.show()

cnn1 = Conv2D(in_channels=1, out_channels=1, kernel_size=(10, 10), stride=(1, 1), padding=(0, 0), name='cnn1')
activation = Tanh()
cnn2 = Conv2D(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), name='cnn2')
fully_connected1 = FC(input_size=15 * 15 * 1 * 1, output_size=1, name='fc1')
# fully_connected2 = FC(input_size=5*5, output_size=1, name='fc2')
sigmoid = Sigmoid()
# layers_list = [
#     ("cnn1", cnn1),
#     ("activation1", activation),
#     ("cnn2", cnn2),
#     ("activation2", activation),
#     ('fc1', fully_connected1),
#     # ('activation3', activation),
#     # ('fc2', fully_connected2),
#     ("sigmoid", sigmoid)
# ]

layers_dic = dict()
layers_dic["cnn1"] = cnn1
layers_dic["activation1"] = activation
layers_dic["cnn2"] = cnn2
layers_dic["activation2"] = activation
layers_dic["fc1"] = fully_connected1
layers_dic["sig"] = sigmoid




model = Model(
    layers_dic,
    BinaryCrossEntropy(),
    optimizer=GD(
        layers_list= layers_dic,
        learning_rate=1 #6.931471807541806
    )
)

model.train(X=images, y=target, epochs=300)
model.save(name="cnn.pkl")
# print(model.predict(images.reshape(1, *images[0].shape)))
print(model.predict(images))