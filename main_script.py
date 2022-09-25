from datetime import datetime
import os
from pathlib import Path

from src.data import load_data_generator
from src.model import compile_model, create_fully_connected_network, fit_model, save_model

import logging
import argparse


def run_model(dtrain, dtest, model_kwargs, compile_kwargs, fit_kwargs, save_path: Path):
    model = create_fully_connected_network(**model_kwargs)
    compile_model(model=model, **compile_kwargs)
    history, training_time = fit_model(model, dtrain, dtest, **fit_kwargs)
    logging.info(f"training_time: {training_time}")

    if not save_path.exists():
        os.mkdir(save_path)
    save_model(model, history, model_kwargs, compile_kwargs, fit_kwargs, save_path)
    logging.info(f"Results saved in folder: {save_path}")


def main(name: str, data: dict, model: dict, compile: dict, fit: dict, run: str):

    model_path = Path(f"./models/{name}")
    if not model_path.exists():
        os.mkdir(model_path)
    log_file = model_path / (run + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    dtrain, dtest = load_data_generator(name=name, **data)
    logging.info(f"Data {name} loaded")
    save_path = model_path / run
    run_model(dtrain, dtest, model, compile, fit, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit neural network with different regularizations")
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    name: str = "cifar10"
    data: dict = dict()
    model: dict = dict(hidden_layers=5, hidden_units=100, input_shape=(32, 32, 3), output_units=10)
    compile: dict = dict()
    fit: dict = dict(epochs=20)
    run = datetime.now().strftime("%Y%m%d%H%M%S")

    main(name=name, data=data, model=model, compile=compile, fit=fit, run=run)
