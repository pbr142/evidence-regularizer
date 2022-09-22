from typing import Optional
import tensorflow as tf
from tensorflow import keras
from time import time

from .regularizer import EvidenceRegularizerLayer


def fully_connected_network(
    input_shape: tuple,
    hidden_layers: int,
    hidden_units: int,
    output_units: int,
    activation=keras.activations.tanh,
    kernel_regularizer=None,
    kernel_initializer=keras.initializers.GlorotUniform,
    kernel_initializer_kwarg: dict = dict(),
    activity_regularizer=None,
    dropout=False,
    dropout_rate=0.5,
    evidence_regularizer_kwargs: Optional[dict] = None,
):
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=input_shape)])
    if dropout:
        model.add(keras.layers.Dropout(rate=dropout_rate))
    layer_args = dict(
        units=hidden_units,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer,
    )
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(**layer_args, kernel_initializer=kernel_initializer(**kernel_initializer_kwarg)))
        if evidence_regularizer_kwargs is not None:
            model.add(EvidenceRegularizerLayer(**evidence_regularizer_kwargs))
    model.add(
        keras.layers.Dense(
            units=output_units, activation="softmax", kernel_initializer=kernel_initializer(**kernel_initializer_kwarg)
        )
    )
    return model


def compile(model, optimizer=keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"]):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


def fit(model, dtrain, dtest, epochs=10, **kwargs):
    history = model.fit(dtrain, epochs=epochs, validation_data=dtest, **kwargs)
    return history


def create_and_fit_fully_connected(dtrain, dtest, model_kwargs=dict(), compile_kwargs=dict(), fit_kwargs=dict()):
    model = fully_connected_network(**model_kwargs)
    compile(model, **compile_kwargs)

    fit_start_time = time()
    history = fit(model, dtrain, dtest, **fit_kwargs)
    fit_end_time = time()

    return model, history, fit_end_time - fit_start_time
