import pandas as pd

from entity_mapper.common import MappingException
from entity_mapper.data.bmus import extract_bm_vols_by_month
from entity_mapper.data.regos import extract_rego_volume_by_month


def appraise_rated_power(generator_profile: dict) -> dict:
    bmus_total_net_capacity = (
        generator_profile["bmus_total_demand_capacity"] + generator_profile["bmus_total_generation_capacity"]
    )
    return dict(
        bmus_total_net_capacity=bmus_total_net_capacity,
        rego_bmu_net_power_ratio=generator_profile["rego_station_dnc_mw"] / bmus_total_net_capacity,
    )


def appraise_energy_volumes(generator_profile: dict, regos: pd.DataFrame) -> dict:
    rego_volume_stats, rego_monthly_volumes = extract_rego_volume_by_month(
        regos,
        generator_profile["rego_station_name"],
        generator_profile["rego_station_dnc_mw"],
    )
    generator_profile.update(rego_volume_stats)

    bmu_volume_stats, bmu_monthly_volumes = extract_bm_vols_by_month(
        generator_profile["bmu_lead_party_id"],
        [bmu["bmu_unit"] for bmu in generator_profile["bmus"]],
        generator_profile["bmus_total_net_capacity"],
    )
    generator_profile.update(bmu_volume_stats)

    rego_and_bmu_monthly_volumes = pd.merge(
        rego_monthly_volumes,
        bmu_monthly_volumes,
        left_index=True,
        right_index=True,
    )
    rego_and_bmu_monthly_volumes["rego_to_bmu_ratio"] = (
        rego_and_bmu_monthly_volumes["GWh"] / rego_and_bmu_monthly_volumes["BM Unit Metered Volume"]
    )
    rego_and_bmu_monthly_volumes.index.name = "start"
    try:
        return dict(
            monthly_energy_volumes=[
                dict(
                    end=str(row["end"]),
                    start=str(row["start"]),
                    bmu_GWh=row["BM Unit Metered Volume"],
                    rego_GWh=row["GWh"],
                    rego_to_bmu_ratio=row["rego_to_bmu_ratio"],
                )
                for _, row in rego_and_bmu_monthly_volumes.reset_index().iterrows()
            ],
            rego_bmu_volume_ratio_median=rego_and_bmu_monthly_volumes["rego_to_bmu_ratio"].median(),
            rego_bmu_volume_ratio_min=rego_and_bmu_monthly_volumes["rego_to_bmu_ratio"].min(),
            rego_bmu_volume_ratio_max=rego_and_bmu_monthly_volumes["rego_to_bmu_ratio"].max(),
        )
    except Exception as e:
        raise MappingException(str(e))
