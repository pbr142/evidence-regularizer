from tensorflow import keras
from functools import partial

from .regularizer import EvidenceRegularizerLayer

hidden_layer = partial(keras.layers.Dense, activation="tanh")
UNITS = [1024, 512, 256, 128, 64]

models = {}


def model(func):
    name = func.__name__.replace("cifar_10_", "")
    models[name] = func()
    return func


@model
def cifar_10_no_regularization():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_no_regularization_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_10():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_10_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_50_10():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=50, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_50_10_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=50, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_10_10():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=10, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_10_10_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=10, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_05():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, strength=0.5, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_05_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, strength=0.5, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_01():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, strength=0.1, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_100_01_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=100, strength=0.1, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_200_01():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=200, strength=0.1, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_200_01_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=200, strength=0.1, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_400_10():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=400, cutoff=0.0))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


@model
def cifar_10_ev_regularization_400_10_bn():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    for units in UNITS:
        model.add(hidden_layer(units=units))
        model.add(EvidenceRegularizerLayer(threshold=400, cutoff=0.0))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model
