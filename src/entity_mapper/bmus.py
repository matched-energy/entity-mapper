import collections
import copy
import json
import os
from pathlib import Path

import click
import pandas as pd


def read_accredited_stations(accredited_stations_dir: Path):
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

    for filename in os.listdir(accredited_stations_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(accredited_stations_dir, filename)
            try:
                df = pd.read_csv(filepath, skiprows=1, names=names)
                df["StationDNC_MW"] = df["StationDNC"].astype(float) / 1e3
                dfs.append(df)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
    return pd.concat(dfs)


def main(bmrs_bm_units: Path) -> pd.DataFrame:
    with open(bmrs_bm_units, "r") as file:
        json_list = json.load(file)
    bmrs_bmus = pd.DataFrame(json_list)
    bmrs_bmus["generationCapacity"] = bmrs_bmus["generationCapacity"].astype(float)
    bmrs_bmus["demandCapacity"] = bmrs_bmus["demandCapacity"].astype(float)
    # , elexon_bm_units: Path):
    # elexon_bmus = pd.read_csv(elexon_bm_units, skiprows=1)
    return bmrs_bmus  # , elexon_bmus


# pd.merge(
#     b[
#         b["leadPartyName"].str.contains("Drax", na=False)
#         & b["productionOrConsumptionFlag"].str.contains("P")
#         & b["generationCapacity"]
#         > 0
#     ],
#     e[e["fuelType"] == "BIOMASS"],
#     left_on="elexonBmUnit",
#     right_on="BM Unit ID",
# )[
#     [
#         col
#         for col in b.columns.append(e.columns)
#         if "orking" not in col
#         and "redit" not in col
#         and "CALF" not in col
#         and "CAIC" not in col
#         and "CAEC" not in col
#         and "Prod/Cons Flag" not in col
#         and "Interconnector" not in col
#         and "FPN Flag" not in col
#         and "BMU Name" not in col
#     ]
# ]

#  b[b["BMU Name"].str.contains("estermost", case=False)][["BM Unit ID", "BMU Name", "Party Name", "Party ID", "GC", "Prod/Cons Flag"]]


@click.command()
@click.option(
    "--bmrs-bm-units",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to BMRS list of BM units (JSON)",
)
@click.option(
    "--elexon-bm-units",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to Elexon list of BM units (CSV)",
)
def main_click(bmrs_bm_units: Path, elexon_bm_units: Path):
    main(bmrs_bm_units, elexon_bm_units)


if __name__ == "__main__":
    main_click()
