import copy
import os
from pathlib import Path

import click
import pandas as pd

import scores.core.supplier_load_by_half_hour


def read_and_agg_vols(vol_by_bsc: Path, bsc: str, bm_ids: list[str]) -> pd.DataFrame:
    vols_df = scores.core.supplier_load_by_half_hour.main(
        vol_by_bsc / Path(bsc),
        Path(f"/tmp/bm_metered_vol_agg_{bsc}.csv"),
        bsc_lead_party_id=bsc,
        bm_regex=None,
        bm_ids=bm_ids,
        group_bms=True,
    )
    return vols_df


def vols_by_month(vols_df: pd.DataFrame) -> pd.DataFrame:
    _vols_df = copy.deepcopy(vols_df)
    _vols_df["Settlement Month"] = vols_df["Settlement Date"].dt.month
    vols_by_month = (
        _vols_df.groupby("Settlement Month")
        .agg(
            {
                "Settlement Date": "first",
                "BM Unit Metered Volume": "sum",
                # "Period BM Unit Balancing Services Volume": "sum",
            }
        )
        .sort_values("Settlement Date")
    ).set_index("Settlement Date")
    vols_by_month["BM Unit Metered Volume"] /= 1e3
    # TODO - adjust? or delete?
    # vols_by_month["Period BM Unit Balancing Services Volume"] /= 1e3
    # vols_by_month["BM Net Vol"] = (
    #     vols_by_month["BM Unit Metered Volume"]
    #     + vols_by_month["Period BM Unit Balancing Services Volume"]
    # )
    return vols_by_month


# @click.command()
# @click.argument(
#     "vol-by-bsc",
#     type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
#     help="Directory containing BSC folders, each containing CSVs",
# )
# @click.argument("bsc", type=str, required=True)
# def main_click(vol_by_bsc: Path):
#     main(vol_by_bsc)


# if __name__ == "__main__":
#     main_click()
