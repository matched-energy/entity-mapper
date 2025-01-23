from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import entity_mapper.utils


def compare_to_expected(mapping_scores: pd.DataFrame, expected_mappings_file: Path) -> pd.DataFrame:
    """Left join"""
    expected_mappings = entity_mapper.utils.from_yaml_file(expected_mappings_file)
    comparisons = []
    for _, row in mapping_scores.iterrows():
        rego_station_name = row["rego_name"]
        bmu_ids = [id.strip() for id in row["bmu_ids"].split(",")]
        expected_bmu_ids = expected_mappings.get(rego_station_name, {}).get("bmu_ids", [])
        comparisons.append(
            dict(
                rego_station_name=rego_station_name,
                bmu_ids=", ".join(bmu_ids),
                expected_bmu_ids=", ".join(expected_bmu_ids),
                verified=(None if not expected_bmu_ids else (set(bmu_ids) == set(expected_bmu_ids))),
            )
        )
    return pd.DataFrame(comparisons)


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
    return summary_row
