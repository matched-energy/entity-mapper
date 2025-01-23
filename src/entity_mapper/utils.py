import logging
import sys

import pandas as pd


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
