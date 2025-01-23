import copy
import json
from pathlib import Path
from typing import Tuple

import elexon_s0142.supplier_load_by_half_hour  # type: ignore # TODO
import pandas as pd

from entity_mapper.common import MappingException


def load_bmus(bmrs_bm_units: Path) -> pd.DataFrame:
    with open(bmrs_bm_units, "r") as file:
        json_list = json.load(file)
    bmrs_bmus = pd.DataFrame(json_list)
    bmrs_bmus["generationCapacity"] = bmrs_bmus["generationCapacity"].astype(float)
    bmrs_bmus["demandCapacity"] = bmrs_bmus["demandCapacity"].astype(float)
    return bmrs_bmus


def read_and_agg_vols(vol_by_bsc: Path, bsc: str, bm_ids: list[str]) -> pd.DataFrame:
    vols_df = elexon_s0142.supplier_load_by_half_hour.main(
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


def extract_bm_vols_by_month(
    lead_party_id: str, bmu_ids: list, bmus_total_net_capacity: float
) -> Tuple[dict, pd.DataFrame]:
    try:
        volumes_df = read_and_agg_vols(
            Path("/Users/jjk/data/2024-12-12-CP2023-all-bscs-s0142/"),
            lead_party_id,
            bmu_ids,
        )
        total_volume = volumes_df["BM Unit Metered Volume"].sum()
        return (
            dict(
                bmu_total_volume=total_volume,
                bmu_capacity_factor=total_volume / (bmus_total_net_capacity * 24 * 365),
                bmu_sampling_months=12,  # NOTE: presumed! TODO: test for this!
            ),
            vols_by_month(volumes_df),
        )
    except Exception as e:
        raise MappingException(f"Failed to extract bm volumes by month {e}")


def get_bmu_list_and_aggregate_properties(bmus: pd.DataFrame) -> dict:
    try:
        assert len(bmus["leadPartyName"].unique()) == 1
        assert len(bmus["leadPartyId"].unique()) == 1
        assert len(bmus["fuelType"].unique()) == 1
    except AssertionError:
        raise MappingException(
            "Expected one lead party and fuel type but got"
            + ", ".join(
                [str(t) for t in bmus[["leadPartyName", "leadPartyId", "fuelType"]].itertuples(index=False, name=None)]
            )
        )
    return dict(
        bmus=[
            dict(
                bmu_unit=bmu["elexonBmUnit"],  # TODO --> bmu_id
                bmu_demand_capacity=bmu["demandCapacity"],
                bmu_generation_capacity=bmu["generationCapacity"],
                bmu_production_or_consumption_flag=bmu["productionOrConsumptionFlag"],
                bmu_transmission_loss_factor=bmu["transmissionLossFactor"],
            )
            for i, bmu in bmus.iterrows()
        ],
        bmus_total_demand_capacity=bmus["demandCapacity"].sum(),
        bmus_total_generation_capacity=bmus["generationCapacity"].sum(),
        bmu_lead_party_name=bmus.iloc[0]["leadPartyName"],
        bmu_lead_party_id=bmus.iloc[0]["leadPartyId"],
        bmu_fuel_type=bmus.iloc[0]["fuelType"],
        lead_party_name_intersection_count=bmus.iloc[0]["leadPartyName_intersection_count"],
        lead_party_name_contiguous_words=bmus.iloc[0]["leadPartyName_contiguous_words"],
    )
