import json
from pathlib import Path

import pandas as pd


def load_bmus(bmrs_bm_units: Path) -> pd.DataFrame:
    with open(bmrs_bm_units, "r") as file:
        json_list = json.load(file)
    bmrs_bmus = pd.DataFrame(json_list)
    bmrs_bmus["generationCapacity"] = bmrs_bmus["generationCapacity"].astype(float)
    bmrs_bmus["demandCapacity"] = bmrs_bmus["demandCapacity"].astype(float)
    return bmrs_bmus
