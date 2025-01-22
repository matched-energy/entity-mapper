import inspect
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scores.common.utils

import entity_mapper.data.regos
from entity_mapper import bm_metered_vol_agg, bmus

REGOS_PATH = (
    "/Users/jjk/Dropbox/data/matched-data/processed/test-data-regos-apr2022-mar2023.csv"
)


class MappingException(Exception):
    """An REGO to BM mapping exception"""

    def __init__(self, message=""):
        super().__init__(message)


# TODO - fix
def current_function_name() -> str:
    return inspect.currentframe().f_code.co_name


def print_warning(function_name: str, warning: str) -> None:
    print(f"!! WARNING !! {function_name}: {warning}")


def load_accredited_stations() -> pd.DataFrame:
    # TODO - rename
    return bmus.read_accredited_stations(
        "/Users/jjk/Library/CloudStorage/Dropbox/data/matched-data/raw/accredited-stations"
    )


def load_bmus() -> pd.DataFrame:
    return bmus.main(
        "/Users/jjk/Dropbox/data/matched-data/raw/bmrs_bm_units-20241211.json"
    )


def lazy_load(
    regos: pd.DataFrame = None,
    accredited_stations: pd.DataFrame = None,
    bmus: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        entity_mapper.data.regos.load(Path(REGOS_PATH)) if regos is None else regos,
        (
            load_accredited_stations()
            if accredited_stations is None
            else accredited_stations
        ),
        load_bmus() if bmus is None else bmus,
    )


def extract_rego_meta_data(
    rego_station_name: str, regos: pd.DataFrame, accredited_stations: pd.DataFrame
) -> dict:
    rego_accreditation_numbers = regos[
        regos["Generating Station / Agent Group"] == rego_station_name
    ]["Accreditation No."].unique()
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


def select_bmu_columns(bmus: pd.DataFrame) -> pd.DataFrame:
    return bmus[
        [
            col
            for col in bmus.columns
            if col
            not in [
                "workingDayCreditAssessmentImportCapability",
                "nonWorkingDayCreditAssessmentImportCapability",
                "workingDayCreditAssessmentExportCapability",
                "nonWorkingDayCreditAssessmentExportCapability",
                "creditQualifyingStatus",
                "gspGroupId",
            ]
        ]
    ]


def filter_on_meta_data_features(bmus: pd.DataFrame, filters: list) -> pd.DataFrame:
    filtered_bmus = bmus.loc[np.logical_and.reduce(filters)]

    try:
        assert len(filtered_bmus) > 0
    except AssertionError:
        warning = "No matching BMUs found"
        raise MappingException(warning)

    return filtered_bmus


def intersection(
    series: pd.Series, value: str, ignore: set = None
) -> (pd.Series, pd.Series):
    if ignore is None:
        ignore = set([])
    intersection_count = series.apply(
        lambda x: (
            0
            if x is None
            else len(
                (
                    set([word.strip("()") for word in x.lower().split()])
                    & set(words(value))
                )
                - ignore
            )
        )
    )
    max_count_filter = (intersection_count > 0) & (
        intersection_count == max(intersection_count)
    )
    return intersection_count, max_count_filter


def filter_on_generation_capacity(
    station_profile: dict, bmus: pd.DataFrame
) -> pd.Series:
    return (
        station_profile["rego_station_dnc_mw"] / 10 < bmus["generationCapacity"]
    ) & (bmus["generationCapacity"] < station_profile["rego_station_dnc_mw"] * 2)


def filter_on_fuel_type(station_profile: dict, bmus: pd.DataFrame) -> pd.Series:
    _, filter = intersection(
        bmus["fuelType"],
        station_profile["rego_station_technology"],
    )
    return filter


def filter_on_name_intersection(
    station_profile: dict, bmus: pd.DataFrame
) -> (pd.Series, pd.Series):
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


def filter_on_name_contiguous(
    station_profile: dict, bmus: pd.DataFrame
) -> (pd.Series, pd.Series):
    lead_party_count = bmus["leadPartyName"].apply(
        lambda x: contiguous_words(station_profile["rego_station_name"], x)
    )
    bmu_count = bmus["bmUnitName"].apply(
        lambda x: contiguous_words(station_profile["rego_station_name"], x)
    )
    max_count = pd.Series(map(max, lead_party_count, bmu_count))
    max_count_filter = (max_count > 0) & (max_count == max(max_count))
    return max_count, max_count_filter


def get_features_and_filters(
    station_profile: dict, bmus: pd.DataFrame
) -> (pd.DataFrame, list):
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


def extract_bmu_meta_data(bmus: pd.DataFrame) -> dict:
    try:
        assert len(bmus["leadPartyName"].unique()) == 1
        assert len(bmus["leadPartyId"].unique()) == 1
        assert len(bmus["fuelType"].unique()) == 1
    except AssertionError:
        raise MappingException(
            f"{current_function_name()} - Expected one lead party and fuel type but got"
            + ", ".join(
                [
                    str(t)
                    for t in bmus[
                        ["leadPartyName", "leadPartyId", "fuelType"]
                    ].itertuples(index=False, name=None)
                ]
            )
        )
    return dict(
        bmus=[
            dict(
                dict(
                    bmu_unit=bmu["elexonBmUnit"],
                    bmu_demand_capacity=bmu["demandCapacity"],
                    bmu_generation_capacity=bmu["generationCapacity"],
                    bmu_production_or_consumption_flag=bmu[
                        "productionOrConsumptionFlag"
                    ],
                    bmu_transmission_loss_factor=bmu["transmissionLossFactor"],
                )
            )
            for i, bmu in bmus.iterrows()
        ],
        bmus_total_demand_capacity=bmus["demandCapacity"].sum(),
        bmus_total_generation_capacity=bmus["generationCapacity"].sum(),
        bmu_lead_party_name=bmus.iloc[0]["leadPartyName"],
        bmu_lead_party_id=bmus.iloc[0]["leadPartyId"],
        bmu_fuel_type=bmus.iloc[0]["fuelType"],
        lead_party_name_intersection_count=bmus.iloc[0][
            "leadPartyName_intersection_count"
        ],
        lead_party_name_contiguous_words=bmus.iloc[0]["leadPartyName_contiguous_words"],
    )


def compare_stated_capacities(generator_profile: dict) -> dict:
    bmus_total_net_capacity = (
        generator_profile["bmus_total_demand_capacity"]
        + generator_profile["bmus_total_generation_capacity"]
    )
    return dict(
        bmus_total_net_capacity=bmus_total_net_capacity,
        rego_bmu_net_power_ratio=generator_profile["rego_station_dnc_mw"]
        / bmus_total_net_capacity,
    )


def extract_rego_volume(
    regos: pd.DataFrame, rego_station_name: str, rego_station_dnc_mw: float
) -> (dict, pd.DataFrame):
    station_regos = regos[
        regos["Generating Station / Agent Group"] == rego_station_name
    ]
    station_regos_by_period = station_regos.groupby(
        ["start", "end", "months_difference"]
    ).agg(dict(GWh="sum"))
    rego_total_volume = station_regos_by_period["GWh"].sum()
    return (
        dict(
            rego_total_volume=rego_total_volume,
            rego_capacity_factor=(
                rego_total_volume
                * 1e3
                / (rego_station_dnc_mw * 24 * 365)  # NOTE: assuming 1 year!
            ),
            rego_sampling_months=12,  # NOTE: presumed!
        ),
        station_regos_by_period.reset_index().set_index("start").sort_index(),
    )


def extract_bm_vols_by_month(
    bmus: pd.DataFrame, bmus_total_net_capacity: float
) -> pd.DataFrame:
    try:
        assert len(bmus["leadPartyId"].unique()) == 1
    except AssertionError:
        raise MappingException(
            f"{current_function_name()} - Expected one leadPartyId but got {list(bmus['leadPartyId'])}"
        )

    try:
        volumes_df = bm_metered_vol_agg.read_and_agg_vols(
            Path("/Users/jjk/data/2024-12-12-CP2023-all-bscs-s0142/"),
            bmus.iloc[0]["leadPartyId"],
            list(bmus["elexonBmUnit"]),
        )
        total_volume = volumes_df["BM Unit Metered Volume"].sum()
        return (
            dict(
                bmu_total_volume=total_volume,
                bmu_capacity_factor=total_volume / (bmus_total_net_capacity * 24 * 365),
                bmu_sampling_months=12,  # NOTE: presumed!
            ),
            bm_metered_vol_agg.vols_by_month(volumes_df),
        )
    except Exception as e:
        raise MappingException(f"Failed to extract bm volumes by month {e}")


def compare_rego_and_bmu_volumes(
    rego_volumes: pd.DataFrame, bmu_volumes: pd.DataFrame
) -> pd.DataFrame:
    merged_df = pd.merge(
        rego_volumes,
        bmu_volumes,
        left_index=True,
        right_index=True,
    )
    merged_df["rego_to_bmu_ratio"] = (
        merged_df["GWh"] / merged_df["BM Unit Metered Volume"]
    )
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
            rego_bmu_volume_ratio_median=volume_comparison[
                "rego_to_bmu_ratio"
            ].median(),
            rego_bmu_volume_ratio_min=volume_comparison["rego_to_bmu_ratio"].min(),
            rego_bmu_volume_ratio_max=volume_comparison["rego_to_bmu_ratio"].max(),
        )
    except Exception as e:
        raise MappingException(e)


def add_score(
    generator_profile: dict,
    column: str,
    score_name: str,
    default: float,
    stat_range,
) -> dict:
    score = {score_name: generator_profile.get(column)}
    for s in stat_range:
        score[f"p({s['lower']}, {s['upper']})"] = (
            s["p"]
            if (s["lower"] <= generator_profile.get(column, default) < s["upper"])
            else 1
        )
    return score


def mapping_score(generator_profile: dict) -> OrderedDict:
    summary_dict = {
        "rego_name": generator_profile.get("rego_station_name"),
        "rego_mw": generator_profile.get("rego_station_dnc_mw"),
        "rego_technology": generator_profile.get("rego_station_technology"),
        "lead_party_name": generator_profile.get("bmu_lead_party_name"),
        "lead_party_id": generator_profile.get("bmu_lead_party_id"),
        "bmu_ids": ", ".join(
            [bmu["bmu_unit"] for bmu in generator_profile.get("bmus", [])]
        ),
        "bmu_fuel_type": generator_profile.get("bmu_fuel_type"),
        "intersection_count": generator_profile.get(
            "lead_party_name_intersection_count"
        ),
    }
    mapping_scores_list = [
        add_score(generator_profile, **score_conf)
        for score_conf in [
            dict(
                column="lead_party_name_contiguous_words",
                score_name="contiguous_words",
                default=0,
                stat_range=[dict(lower=3, upper=float("inf"), p=0.1)],
            ),
            dict(
                column="rego_bmu_volume_ratio_median",
                score_name="volume_ratio_p50",
                default=0,
                stat_range=[
                    dict(lower=0.7, upper=1.05, p=0.5),
                    dict(lower=0.9, upper=1.05, p=0.1),
                ],
            ),
            dict(
                column="rego_bmu_volume_ratio_min",
                score_name="volume_ratio_min",
                default=0,
                stat_range=[dict(lower=0.1, upper=1.0, p=0.5)],
            ),
            dict(
                column="rego_bmu_volume_ratio_max",
                score_name="volume_ratio_max",
                default=0,
                stat_range=[dict(lower=0.5, upper=1.1, p=0.5)],
            ),
            dict(
                column="rego_bmu_net_power_ratio",
                score_name="power_ratio",
                default=0,
                stat_range=[
                    dict(lower=0.5, upper=2, p=0.5),
                    dict(lower=0.95, upper=1.05, p=0.1),
                ],
            ),
        ]
    ]
    row = pd.DataFrame(
        [
            summary_dict
            | {k: v for score in mapping_scores_list for k, v in score.items()}
        ]
    )
    row["p"] = row[[col for col in row.columns if "p(" in col]].prod(axis=1)
    return row


def main_individual(
    rego_station_name: str,
    regos: pd.DataFrame = None,
    accredited_stations: pd.DataFrame = None,
    bmus: pd.DataFrame = None,
    expected_mappings_file: Path = None,
):
    regos, accredited_stations, bmus = lazy_load(regos, accredited_stations, bmus)
    expected_mapping = (
        scores.common.utils.from_yaml_file(expected_mappings_file)
        if expected_mappings_file
        else {}
    ).get(rego_station_name, dict(bmu_ids=[], override=False))

    generator_profile = {}
    matching_bmus = None
    try:
        generator_profile.update(
            extract_rego_meta_data(rego_station_name, regos, accredited_stations)
        )

        if expected_mapping["bmu_ids"] and expected_mapping.get("override"):
            matching_bmus = bmus[bmus["elexonBmUnit"].isin(expected_mapping["bmu_ids"])]
            features, filters = get_features_and_filters(
                generator_profile, matching_bmus
            )
            matching_bmus = pd.concat([matching_bmus, features], axis=1)
        else:
            features, filters = get_features_and_filters(generator_profile, bmus)
            matching_bmus = pd.concat([bmus, features], axis=1)
            matching_bmus = filter_on_meta_data_features(matching_bmus, filters)
        matching_bmus = select_bmu_columns(matching_bmus)
        generator_profile.update(extract_bmu_meta_data(matching_bmus))

        generator_profile.update(compare_stated_capacities(generator_profile))

        rego_volume_stats, rego_volumes = extract_rego_volume(
            regos,
            generator_profile["rego_station_name"],
            generator_profile["rego_station_dnc_mw"],
        )
        generator_profile.update(rego_volume_stats)

        bmu_volume_stats, bmu_volumes = extract_bm_vols_by_month(
            matching_bmus, generator_profile["bmus_total_net_capacity"]
        )
        generator_profile.update(bmu_volume_stats)

        volume_comparison = compare_rego_and_bmu_volumes(rego_volumes, bmu_volumes)
        generator_profile.update(parse_volume_comparison(volume_comparison))
    except MappingException as e:
        print("\n### EXCEPTION\n" + str(e))
        print("\n### GENERATOR PROFILE\n" + str(generator_profile))
    print(scores.common.utils.to_yaml_text(generator_profile))
    return mapping_score(generator_profile)


def main_range(
    start: int,
    stop: int,
    regos: pd.DataFrame = None,
    accredited_stations: pd.DataFrame = None,
    bmus: pd.DataFrame = None,
    expected_mappings_file: Path = None,
) -> pd.DataFrame:
    regos, accredited_stations, bmus = lazy_load(regos, accredited_stations, bmus)
    regos_by_station = entity_mapper.data.regos.groupby_station(regos)
    station_summaries = []
    for i in range(start, stop):
        station_summaries.append(
            main_individual(
                regos_by_station.iloc[i]["Generating Station / Agent Group"],
                regos,
                accredited_stations,
                bmus,
                expected_mappings_file,
            )
        )
    return pd.concat(station_summaries)


def compare_to_expected(
    mapping_scores: pd.DataFrame, expected_mappings_file: Path
) -> pd.DataFrame:
    """Left join"""
    expected_mappings = scores.common.utils.from_yaml_file(expected_mappings_file)
    comparisons = []
    for _, row in mapping_scores.iterrows():
        rego_station_name = row["rego_name"]
        bmu_ids = [id.strip() for id in row["bmu_ids"].split(",")]
        expected_bmu_ids = expected_mappings.get(rego_station_name, {}).get(
            "bmu_ids", []
        )
        comparisons.append(
            dict(
                rego_station_name=rego_station_name,
                bmu_ids=", ".join(bmu_ids),
                expected_bmu_ids=", ".join(expected_bmu_ids),
                verified=(
                    None
                    if not expected_bmu_ids
                    else (set(bmu_ids) == set(expected_bmu_ids))
                ),
            )
        )
    return pd.DataFrame(comparisons)
    return pd.DataFrame(comparisons)
