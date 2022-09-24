import pickle
from pathlib import Path
from time import time
from typing import Optional

from tensorflow import keras

from .regularizer import EvidenceRegularizerLayer


def create_fully_connected_network(
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


def compile_model(model, optimizer=keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"]):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


def fit_model(model, dtrain, dtest, epochs=10, **kwargs):
    fit_start_time = time()
    history = model.fit(dtrain, epochs=epochs, validation_data=dtest, **kwargs)
    fit_end_time = time()
    return history.history, fit_end_time - fit_start_time


def create_and_fit_fully_connected(dtrain, dtest, model_kwargs=dict(), compile_kwargs=dict(), fit_kwargs=dict()):
    model = create_fully_connected_network(**model_kwargs)
    compile_model(model, **compile_kwargs)
    history, training_time = fit_model(model, dtrain, dtest, **fit_kwargs)
    return model, history, training_time


def save_model(model, history, model_kwargs, compile_kwargs, fit_kwargs, path: Path):
    model_file = path / "model.pb"
    model.save(model_file)
    history_file = path / "history.pickle"
    with open(history_file, "wb") as f:
        pickle.dump(history, f, protocol=5)
    cfg = dict(model=model_kwargs, compile=compile_kwargs, fit=fit_kwargs)
    cfg_file = path / "cfg.yml"
    with open(cfg_file, "wb") as f:
        pickle.dump(cfg, f, protocol=0)


def load_model(path: Path):
    model_file = path / "model.pb"
    model = keras.models.load_model(model_file)
    history_file = path / "history.pickle"
    with open(history_file, "rb") as f:
        history = pickle.load(f)
    cfg_file = path / "cfg.yml"
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)
    return model, history, cfg["model"], cfg["compile"], cfg["fit"]
