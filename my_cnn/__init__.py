"""my_cnn: a small convolutional neural network library written from scratch in NumPy.

The API is modeled on Keras::

    import my_cnn
    from my_cnn import layers

    model = my_cnn.Model()
    model.add(layers.Input((28, 28, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation_func='softmax'))
    model.compile(optimizer='momentum', loss='crossentropy', learning_rate=0.02)
"""

from . import layers, losses, optimizers
from .model import Model
