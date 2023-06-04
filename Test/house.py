import numpy as np
import pandas as pd
from  typing import Dict, Set,List

from activations import LinearActivation, ReLU, Tanh
from layers.fully_connected_test import FC
from losses.meansquarederror import MeanSquaredError
from model import Model
from optimizers.gradientdescent import GD

data = pd.read_csv("../datasets/california_houses_price/california_housing_train.csv")

X_train, y_train = data.drop("median_house_value", axis=1), data[["median_house_value"]]


X_train_arr = np.array(X_train)


y_train_arr = np.array(y_train)
max_of_y = np.max(y_train_arr)

y_train_arr = y_train_arr / max_of_y


fc_1 = FC(8, 100, "fc1")
fc_2 = FC(input_size=100, output_size=60, name='fc2')
# fc_3 = FC(input_size=80, output_size=60, name='fc3')
fc_3 = FC(input_size=60, output_size=1, name='fc3')

# In[14]:


linear = LinearActivation()
relu = ReLU()

# In[15]:

#
# layers_list=[
#     ('fc1', fc_1),
#     ('linear', Tanh()),
#     ('fc2', fc_2),
#     ('linear', Tanh()),
#     # ('fc3', fc_3),
#     # ('linear', ReLU()),
#     ('fc4', fc_4),
#     ('linear', LinearActivation()),
# ]


layers = dict()
layers['fc1'] = fc_1
layers['Thanh1'] = Tanh()
layers['fc2'] = fc_2
layers['Tanh2'] = Tanh()
layers['fc3'] = fc_3
layers['RELU3'] = ReLU()


# In[16]:


model = Model(
    layers,
    criterion=MeanSquaredError(),
    optimizer=GD(layers_list=layers, learning_rate=0.01)
)


# In[17]:


model.train(X_train_arr, y_train_arr, epochs=1000)

# print("**********")
# print("predicted values:")
# print(model.predict(X_train_arr[:10]) * max_of_y)
# print("**********")
# print("actual values:")
model.save(name="regressor.pkl")
print("error: ", MeanSquaredError().compute(y_pred=model.predict(X_train_arr) * max_of_y, y_true=y_train_arr * max_of_y))



# In[ ]:



