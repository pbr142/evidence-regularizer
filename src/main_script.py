from datetime import datetime
import os
from pathlib import Path

from .data import load_data_generator
from .model import compile_model, create_fully_connected_network, fit_model, save_model
from .kwargs import get_config, run_model_kwargs, TYPES

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


def main(name: str, min_evidence_fraction: float = 0.1, batch_size: int = 1000, cutoff: float = 0.0):

    model_path = Path(f"./models/{name}")
    if not model_path.exists():
        os.mkdir(model_path)
    run = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = model_path / (run + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    data_kwargs: dict = dict(name=name, batch_size=batch_size)
    dtrain, dtest = load_data_generator(**data_kwargs)
    logging.info(f"Data {name} loaded")
    model_kwargs, compile_kwargs, fit_kwargs = get_config(name=name)

    save_path = model_path / run

    for type in TYPES:
        evidence_regularizer_kwargs = dict(threshold=min_evidence_fraction * batch_size, cutoff=cutoff)
        model_kwargs = run_model_kwargs(type, model_kwargs, evidence_regularizer_kwargs)
        run_model(dtrain, dtest, model_kwargs, compile_kwargs, fit_kwargs, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit neural network with different regularizations")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--min_evidence_fraction", type=float, required=False, default=0.1)
    parser.add_argument("--batch_size", type=int, required=False, default=1000)
    parser.add_argument("--cutoff", type=float, required=False, default=0.0)

    args = parser.parse_args()

    main(**vars(args))
