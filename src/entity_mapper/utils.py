import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from yaml import Dumper, ScalarNode


def select_columns(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    return df[[col for col in df.columns if col not in exclude]]


def get_logger(name: str, level: str = "debug") -> logging.Logger:
    logger = logging.getLogger(name)

    # Set level
    levels = dict(
        debug=logging.DEBUG,
        info=logging.INFO,
        warning=logging.WARNING,
        error=logging.ERROR,
        critical=logging.CRITICAL,
    )
    try:
        logger.setLevel(levels[level])
    except KeyError:
        raise KeyError(f"Expect level to be in {list(levels.keys())}")

    # Set handler and formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def float_representer(dumper: Dumper, value: Union[float, np.float32, np.float64]) -> ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", format(value, ".2f"))


yaml.add_representer(float, float_representer)
yaml.add_representer(np.float32, float_representer)
yaml.add_representer(np.float64, float_representer)


def from_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=yaml.FullLoader)


def from_yaml_file(path: Path) -> Any:
    with open(path, "r") as file:
        return from_yaml_text(file.read())


def to_yaml_text(obj: Any) -> Optional[str]:
    return yaml.dump(obj, default_flow_style=False)


def to_yaml_file(obj: Any, path: Path) -> None:
    with open(path, "w") as file:
        file.write(to_yaml_text(obj))
