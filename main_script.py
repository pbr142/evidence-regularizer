from datetime import datetime
import os
from pathlib import Path

from src.data import load_data_generator
from src.model import compile_model, create_fully_connected_network, fit_model, save_model
from src.config import load_config, parse_config, parse_experiments

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
    log_file = model_path / (run + ".log")
    log_path = log_file.resolve().parent
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO)

    dtrain, dtest = load_data_generator(name=name, **data)
    logging.info(f"Data {name} loaded")
    save_path = model_path / run
    save_path.mkdir(parents=True, exist_ok=True)
    run_model(dtrain, dtest, model, compile, fit, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit neural network with different regularizations")
    parser.add_argument("-c", "--config", type=str, default="default.yml")
    parser.add_argument("-r", "--run", type=str, default=None)
    args = parser.parse_args()

    config = args.config
    run = datetime.now().strftime("%Y%m%d%H%M%S") if args.run is None else args.run

    cfg = load_config(config)
    name, data, model, compile, fit, experiments = parse_config(cfg)
    if experiments is not None:
        experiments = parse_experiments(data, model, compile, fit, experiments)
        for experiment, ex_cfg in experiments.items():
            run_experiment = run + "/" + experiment
            main(name=name, **ex_cfg, run=run_experiment)
    else:
        main(name, data, model, compile, fit, run=run)
