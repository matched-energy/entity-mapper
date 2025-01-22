import os
from pathlib import Path

import pandas as pd


def load_accredited_stations(accredited_stations_dir: Path) -> pd.DataFrame:
    names = [
        "AccreditationNumber",
        "Status",
        "GeneratingStation",
        "Scheme",
        "StationDNC",
        "Country",
        "Technology",
        "ContractType",
        "AccreditationDate",
        "CommissionDate",
        "Organisation",
        "OrganisationContactAddress",
        "OrganisationContactFax",
        "GeneratingStationAddress",
    ]
    dfs = []

    with os.scandir(accredited_stations_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".csv"):
                filepath = Path(entry.path)
                try:
                    df = pd.read_csv(filepath, skiprows=1, names=names)
                    df["StationDNC_MW"] = df["StationDNC"].astype(float) / 1e3
                    dfs.append(df)
                except ValueError as e:
                    print(f"Skipping {entry.name}: {e}")

    return pd.concat(dfs)
