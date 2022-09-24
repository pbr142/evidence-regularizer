from datetime import datetime
import os
from pathlib import Path

from src.data import load_data_generator
from src.model import compile_model, create_fully_connected_network, fit_model, save_model

import logging

data_kwargs = dict(name="cifar100", batch_size=1000, seed=1234)

evidence_regularizer_kwargs = dict(threshold=10, cutoff=0)
model_kwargs = dict(
    input_shape=(32, 32, 3),
    hidden_layers=5,
    hidden_units=10,
    output_units=100,
    evidence_regularizer_kwargs=evidence_regularizer_kwargs,
)
compile_kwargs: dict = dict()
fit_kwargs = dict(epochs=100)


def main():
    model_path = Path(f"./models/{data_kwargs['name']}")
    if not model_path.exists():
        os.mkdir(model_path)
    run = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = model_path / (run + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    logging.info(f"data_kwargs: {data_kwargs}")
    logging.info(f"model_kwargs: {model_kwargs}")
    logging.info(f"compile_kwargs: {compile_kwargs}")
    logging.info(f"compile_kwargs: {fit_kwargs}")

    dtrain, dtest = load_data_generator(**data_kwargs)
    model = create_fully_connected_network(**model_kwargs)
    compile_model(model=model, **compile_kwargs)
    history, training_time = fit_model(model, dtrain, dtest, **fit_kwargs)

    logging.info(f"training_time: {training_time}")

    save_path = model_path / run
    if not model_path.exists():
        os.mkdir(save_path)
    save_model(model, history, model_kwargs, compile_kwargs, fit_kwargs, save_path)
    logging.info(f"Results saved in folder: {save_path}")


if __name__ == "__main__":
    main()
