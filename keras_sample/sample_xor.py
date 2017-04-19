from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.optimizers import SGD
#from keras.utils.visualize_util import plot

#from binary_vis_tools import progress_plot

import numpy as np

# Create the neural network model
model = Sequential([
    Dense(2, kernel_initializer='random_normal', input_dim=2),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid')])

# Activation: sigmoid, relu, tanh, softmax
# optimizer: SGD, RMSprop
# loss: mean_squared_error, binary_crossentropy

# Compile the model with an optimiser
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

# Create training cases for an XOR function
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
model.load_weights("sample_xor.hdf")
model.fit(x, y, epochs=6666)
print model.get_weights()
model.save("sample_xor.hdf")
print model.predict(x)
#progress_plot(model, x, y, bottomLeft, topRight, epochList)