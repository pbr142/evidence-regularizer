from tensorflow import keras
from functools import partial

from .regularizer import EvidenceRegularizerLayer

hidden_layer = partial(keras.layers.Dense, activation="tanh")

models = {}
depth = 30


def model(func):
    for i in range(depth):
        name = func.__name__.replace("cifar_10_", "") + f"_{i}"
        models[name] = func(layers=i)
    return func


@model
def cifar_10_no_regularization(layers, units=128):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for i in range(layers):
        model.add(hidden_layer(units=units))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization(layers, units=128):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for i in range(layers):
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model
