import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import scores.common.utils

import entity_mapper.utils
from entity_mapper.common import MappingException
from entity_mapper.data.bmus import extract_bm_vols_by_month, get_bmu_list_and_aggregate_properties, load_bmus
from entity_mapper.data.regos import (
    extract_rego_volume,
    get_generator_profile,
    groupby_regos_by_station,
    load_accredited_stations,
    load_regos,
)
from entity_mapper.match_meta_data import apply_bmu_match_filters, define_bmu_match_features_and_filters

LOGGER = entity_mapper.utils.get_logger("entity_mapper")


def appraise_rated_power(generator_profile: dict) -> dict:
    bmus_total_net_capacity = (
        generator_profile["bmus_total_demand_capacity"] + generator_profile["bmus_total_generation_capacity"]
    )
    return dict(
        bmus_total_net_capacity=bmus_total_net_capacity,
        rego_bmu_net_power_ratio=generator_profile["rego_station_dnc_mw"] / bmus_total_net_capacity,
    )


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


def get_matching_bmus(generator_profile: dict, bmus: pd.DataFrame, expected_mapping: dict) -> pd.DataFrame:
    # Determine if should rate expected BMUs or search over all BMUs
    expected_overrides = expected_mapping["bmu_ids"] and expected_mapping.get("override")
    bmus_to_search = (
        bmus[bmus["elexonBmUnit"].isin(expected_mapping["bmu_ids"])] if expected_overrides else copy.deepcopy(bmus)
    )

    # Define matching features and filters
    bmu_match_features, bmu_match_filters = define_bmu_match_features_and_filters(generator_profile, bmus_to_search)
    bmus_to_search = bmus_to_search.join(bmu_match_features, how="outer")

    # Return expected / filtered BMUs with matching
    matching_bmus = bmus_to_search if expected_overrides else apply_bmu_match_filters(bmus_to_search, bmu_match_filters)
    return entity_mapper.utils.select_columns(
        matching_bmus,
        exclude=[
            "workingDayCreditAssessmentImportCapability",
            "nonWorkingDayCreditAssessmentImportCapability",
            "workingDayCreditAssessmentExportCapability",
            "nonWorkingDayCreditAssessmentExportCapability",
            "creditQualifyingStatus",
            "gspGroupId",
        ],
    )


def get_p_values_for_metric(
    metric_name: str,
    value: Any,
    p_value_ranges: List[Dict],
) -> dict:
    p_val_dict = OrderedDict({metric_name: value})
    for s in p_value_ranges:
        p_val_dict[f"p({s['lower']}, {s['upper']})"] = s["p"] if (s["lower"] <= value < s["upper"]) else 1
    return p_val_dict


def get_p_values_for_all_metrics(generator_profile: dict) -> List:
    return [
        get_p_values_for_metric(
            metric_name="contiguous_words",
            value=generator_profile.get("lead_party_name_contiguous_words", 0),
            p_value_ranges=[dict(lower=3, upper=float("inf"), p=0.1)],
        ),
        get_p_values_for_metric(
            metric_name="volume_ratio_p50",
            value=generator_profile.get("rego_bmu_volume_ratio_median", 0),
            p_value_ranges=[
                dict(lower=0.7, upper=1.05, p=0.5),
                dict(lower=0.9, upper=1.05, p=0.1),
            ],
        ),
        get_p_values_for_metric(
            metric_name="volume_ratio_min",
            value=generator_profile.get("rego_bmu_volume_ratio_min", 0),
            p_value_ranges=[dict(lower=0.1, upper=1.0, p=0.5)],
        ),
        get_p_values_for_metric(
            metric_name="volume_ratio_max",
            value=generator_profile.get("rego_bmu_volume_ratio_max", 0),
            p_value_ranges=[dict(lower=0.5, upper=1.1, p=0.5)],
        ),
        get_p_values_for_metric(
            metric_name="power_ratio",
            value=generator_profile.get("rego_bmu_net_power_ratio", 0),
            p_value_ranges=[
                dict(lower=0.5, upper=2, p=0.5),
                dict(lower=0.95, upper=1.05, p=0.1),
            ],
        ),
    ]


def summarise_mapping_and_mapping_strength(generator_profile: dict) -> pd.DataFrame:
    mapping_summary = {
        "rego_name": generator_profile.get("rego_station_name"),
        "rego_mw": generator_profile.get("rego_station_dnc_mw"),
        "rego_technology": generator_profile.get("rego_station_technology"),
        "lead_party_name": generator_profile.get("bmu_lead_party_name"),
        "lead_party_id": generator_profile.get("bmu_lead_party_id"),
        "bmu_ids": ", ".join([bmu["bmu_unit"] for bmu in generator_profile.get("bmus", [])]),
        "bmu_fuel_type": generator_profile.get("bmu_fuel_type"),
        "intersection_count": generator_profile.get("lead_party_name_intersection_count"),
    }
    mapping_strength = {
        k: v for p_val_dict in get_p_values_for_all_metrics(generator_profile) for k, v in p_val_dict.items()
    }
    # A single row that summarises the mapping and mapping strength
    summary_row = pd.DataFrame([mapping_summary | mapping_strength])
    # An aggregate p-value that is the product of all others
    summary_row["p"] = summary_row[[col for col in summary_row.columns if "p(" in col]].prod(axis=1)
    return summary_row


def map_station(
    rego_station_name: str,
    regos: pd.DataFrame,
    accredited_stations: pd.DataFrame,
    bmus: pd.DataFrame,
    expected_mappings: Optional[dict] = None,
) -> pd.DataFrame:
    if not expected_mappings:
        expected_mappings = {}
    expected_mapping = expected_mappings.get(rego_station_name, dict(bmu_ids=[], override=False))

    generator_profile = {}
    matching_bmus = None
    try:
        # Get details of a REGO generator
        generator_profile.update(get_generator_profile(rego_station_name, regos, accredited_stations))

        # Add matching BMUs
        matching_bmus = get_matching_bmus(generator_profile, bmus, expected_mapping)
        generator_profile.update(get_bmu_list_and_aggregate_properties(matching_bmus))

        # Appraise rated power
        generator_profile.update(appraise_rated_power(generator_profile))

        # Appraise energy volumes
        generator_profile.update(appraise_energy_volumes(generator_profile, regos))

    except MappingException as e:
        LOGGER.warning(str(e) + str(generator_profile))
    LOGGER.debug(scores.common.utils.to_yaml_text(generator_profile))
    return summarise_mapping_and_mapping_strength(generator_profile)


def map_station_range(
    start: int,
    stop: int,
    regos: pd.DataFrame,
    accredited_stations: pd.DataFrame,
    bmus: pd.DataFrame,
    expected_mappings: Optional[dict] = None,
) -> pd.DataFrame:
    regos_by_station = groupby_regos_by_station(regos)
    station_summaries = []
    for i in range(start, stop):
        station_summaries.append(
            map_station(
                regos_by_station.iloc[i]["Generating Station / Agent Group"],
                regos,
                accredited_stations,
                bmus,
                expected_mappings,
            )
        )
    return pd.concat(station_summaries)


def main(
    start: int,
    stop: int,
    regos_path: Path,
    accredited_stations_dir: Path,
    bmus_path: Path,
    expected_mappings_file: Optional[Path] = None,
) -> pd.DataFrame:
    return map_station_range(
        start=start,
        stop=stop,
        regos=load_regos(regos_path),
        accredited_stations=load_accredited_stations(accredited_stations_dir),
        bmus=load_bmus(bmus_path),
        expected_mappings=(
            scores.common.utils.from_yaml_file(expected_mappings_file) if expected_mappings_file else {}
        ),
    )
