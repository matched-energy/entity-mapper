import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scores.common.utils

import entity_mapper.utils
from entity_mapper.common import MappingException
from entity_mapper.data.accredited_stations import load_accredited_stations
from entity_mapper.data.bmus import extract_bm_vols_by_month, load_bmus
from entity_mapper.data.regos import extract_rego_volume, groupby_regos_by_station, load_regos

LOGGER = entity_mapper.utils.get_logger("entity_mapper")


def get_generator_profile(rego_station_name: str, regos: pd.DataFrame, accredited_stations: pd.DataFrame) -> dict:
    rego_accreditation_numbers = regos[regos["Generating Station / Agent Group"] == rego_station_name][
        "Accreditation No."
    ].unique()
    try:
        assert len(rego_accreditation_numbers) == 1
    except Exception:
        raise MappingException(
            f"Found multiple accreditation numbers for {rego_station_name}: {rego_accreditation_numbers}"
        )

    rego_accreditation_number = rego_accreditation_numbers[0]
    accredited_station = accredited_stations[
        (accredited_stations["AccreditationNumber"] == rego_accreditation_number)
        & (accredited_stations["Scheme"] == "REGO")
    ]
    try:
        assert len(accredited_station) == 1
    except Exception:
        raise MappingException(
            f"Expected 1 accredited_station for {rego_accreditation_numbers} but found"
            + f"{list(accredited_station['GeneratingStation'][:5])}"
        )

    return dict(
        {
            "rego_station_name": rego_station_name,
            "rego_accreditation_number": rego_accreditation_number,
            "rego_station_dnc_mw": accredited_station.iloc[0]["StationDNC_MW"],
            "rego_station_technology": accredited_station.iloc[0]["Technology"],
        }
    )


def words(name: str) -> list:
    return [] if name is None else [word.strip("()") for word in name.lower().split()]


def contiguous_words(l_name: str, r_name: str) -> int:
    count = 0
    for l, r in zip(words(l_name), words(r_name)):
        if l == r:
            count += 1
        else:
            break
    return count


def apply_bmu_match_filters(bmus: pd.DataFrame, filters: list) -> pd.DataFrame:
    filtered_bmus = bmus.loc[np.logical_and.reduce(filters)]

    try:
        assert len(filtered_bmus) > 0
    except AssertionError:
        warning = "No matching BMUs found"
        raise MappingException(warning)

    return filtered_bmus


def intersection(series: pd.Series, value: str, ignore: set = None) -> (pd.Series, pd.Series):
    if ignore is None:
        ignore = set([])
    intersection_count = series.apply(
        lambda x: (
            0
            if x is None
            else len((set([word.strip("()") for word in x.lower().split()]) & set(words(value))) - ignore)
        )
    )
    max_count_filter = (intersection_count > 0) & (intersection_count == max(intersection_count))
    return intersection_count, max_count_filter


def filter_on_generation_capacity(station_profile: dict, bmus: pd.DataFrame) -> pd.Series:
    return (station_profile["rego_station_dnc_mw"] / 10 < bmus["generationCapacity"]) & (
        bmus["generationCapacity"] < station_profile["rego_station_dnc_mw"] * 2
    )


def filter_on_fuel_type(station_profile: dict, bmus: pd.DataFrame) -> pd.Series:
    _, filter = intersection(
        bmus["fuelType"],
        station_profile["rego_station_technology"],
    )
    return filter


def filter_on_name_intersection(station_profile: dict, bmus: pd.DataFrame) -> (pd.Series, pd.Series):
    lead_party_count, _ = intersection(
        bmus["leadPartyName"],
        station_profile["rego_station_name"],
        ignore=set(["wind", "farm", "windfarm", "limited", "ltd"]),
    )
    bmu_intersection_count, _ = intersection(
        bmus["bmUnitName"],
        station_profile["rego_station_name"],
        ignore=set(["wind", "farm", "windfarm", "limited", "ltd"]),
    )
    max_count = pd.Series(map(max, lead_party_count, bmu_intersection_count))
    max_count_filter = (max_count > 0) & (max_count == max(max_count))
    return max_count, max_count_filter


def filter_on_name_contiguous(station_profile: dict, bmus: pd.DataFrame) -> (pd.Series, pd.Series):
    lead_party_count = bmus["leadPartyName"].apply(lambda x: contiguous_words(station_profile["rego_station_name"], x))
    bmu_count = bmus["bmUnitName"].apply(lambda x: contiguous_words(station_profile["rego_station_name"], x))
    max_count = pd.Series(map(max, lead_party_count, bmu_count))
    max_count_filter = (max_count > 0) & (max_count == max(max_count))
    return max_count, max_count_filter


def rate_bmu_match(station_profile: dict, bmus: pd.DataFrame) -> (pd.DataFrame, list):
    features = pd.DataFrame(index=bmus.index)
    filters = []

    filter = filter_on_generation_capacity(station_profile, bmus)
    filters.append(filter)

    filter = filter_on_fuel_type(station_profile, bmus)
    filters.append(filter)

    feature, filter = filter_on_name_intersection(station_profile, bmus)
    features["leadPartyName_intersection_count"] = feature  # TODO
    filters.append(filter)

    feature, filter = filter_on_name_contiguous(station_profile, bmus)
    features["leadPartyName_contiguous_words"] = feature  # TODO
    filters.append(filter)

    return features, filters


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
    bmu_match_features, bmu_match_filters = rate_bmu_match(generator_profile, bmus_to_search)
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
