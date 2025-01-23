import pandas as pd

from entity_mapper.common import MappingException
from entity_mapper.data.bmus import extract_bm_vols_by_month
from entity_mapper.data.regos import extract_rego_volume


def appraise_rated_power(generator_profile: dict) -> dict:
    bmus_total_net_capacity = (
        generator_profile["bmus_total_demand_capacity"] + generator_profile["bmus_total_generation_capacity"]
    )
    return dict(
        bmus_total_net_capacity=bmus_total_net_capacity,
        rego_bmu_net_power_ratio=generator_profile["rego_station_dnc_mw"] / bmus_total_net_capacity,
    )


def compare_rego_and_bmu_volumes(rego_volumes: pd.DataFrame, bmu_volumes: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(
        rego_volumes,
        bmu_volumes,
        left_index=True,
        right_index=True,
    )
    merged_df["rego_to_bmu_ratio"] = merged_df["GWh"] / merged_df["BM Unit Metered Volume"]
    merged_df.index.name = "start"
    return merged_df


def parse_volume_comparison(volume_comparison: pd.DataFrame) -> dict:
    try:
        return dict(
            volume_periods=[
                dict(
                    end=str(row["end"]),
                    start=str(row["start"]),
                    bmu_GWh=row["BM Unit Metered Volume"],
                    rego_GWh=row["GWh"],
                    rego_to_bmu_ratio=row["rego_to_bmu_ratio"],
                )
                for _, row in volume_comparison.reset_index().iterrows()
            ],
            rego_bmu_volume_ratio_median=volume_comparison["rego_to_bmu_ratio"].median(),
            rego_bmu_volume_ratio_min=volume_comparison["rego_to_bmu_ratio"].min(),
            rego_bmu_volume_ratio_max=volume_comparison["rego_to_bmu_ratio"].max(),
        )
    except Exception as e:
        raise MappingException(e)


def appraise_energy_volumes(generator_profile: dict, regos: pd.DataFrame) -> dict:
    rego_volume_stats, rego_volumes = extract_rego_volume(
        regos,
        generator_profile["rego_station_name"],
        generator_profile["rego_station_dnc_mw"],
    )
    generator_profile.update(rego_volume_stats)

    bmu_volume_stats, bmu_volumes = extract_bm_vols_by_month(
        generator_profile["bmu_lead_party_id"],
        [bmu["bmu_unit"] for bmu in generator_profile["bmus"]],
        generator_profile["bmus_total_net_capacity"],
    )
    generator_profile.update(bmu_volume_stats)

    volume_comparison = compare_rego_and_bmu_volumes(rego_volumes, bmu_volumes)
    return parse_volume_comparison(volume_comparison)
