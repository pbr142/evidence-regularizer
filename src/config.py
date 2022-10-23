from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union
from pathlib import Path

CONFIG_PATH = Path("./config").resolve()


def load_config(config_name: str) -> Union[DictConfig, ListConfig]:
    if not config_name.endswith(".yml"):
        config_name = config_name + ".yml"
    config_path = CONFIG_PATH / config_name
    return OmegaConf.load(config_path)


def parse_config(config: DictConfig):
    name = config.name
    data = config.get("data", {})
    model = config.model
    compile = config.get("compile", {})
    fit = config.get("fit", {})
    experiments = config.get("experiments", None)

    return name, data, model, compile, fit, experiments


def parse_experiments(data, model, compile, fit, experiments):
    for name, experiment in experiments.items():
        experiment.data = OmegaConf.merge(data, experiment.get("data", {}))
        experiment.model = OmegaConf.merge(model, experiment.get("model", {}))
        experiment.compile = OmegaConf.merge(compile, experiment.get("compile", {}))
        experiment.fit = OmegaConf.merge(fit, experiment.get("fit", {}))
    return experiments
