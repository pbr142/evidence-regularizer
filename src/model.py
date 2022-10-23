from itertools import product
from pathlib import Path
from typing import Optional
import pandas as pd


def load_history(path):
    history_file = path / 'history.csv'
    df_history = pd.read_csv(history_file, index_col='epoch')
    return df_history

def parse_patterns(patterns: list) -> list:
    for i, p in enumerate(patterns):
        if not isinstance(p, list):
            patterns[i] = [p]
    
    return list(product(*patterns))

def load_histories(path: Path, patterns: Optional[list] = None) -> pd.DataFrame:
    if patterns is None:
        histories = {experiment.name: load_history(experiment).melt(ignore_index=False) for experiment in path.iterdir()}
    else:
        patterns = parse_patterns(patterns)
        histories = dict()
        for experiment in path.iterdir():
            if any([all([p in experiment.name for p in pattern]) for pattern in patterns]):
                histories[experiment.name] = load_history(experiment).melt(ignore_index=False)
    
    df_histories = pd.concat(histories, names=['experiment', 'epoch'])
    df_histories.reset_index(inplace=True)

    return df_histories

