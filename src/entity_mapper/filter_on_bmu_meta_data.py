import numpy as np
import pandas as pd


################################################################################
# HELPER FUNCTIONS
################################################################################
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


################################################################################
# FILTERS
################################################################################
def filter_on_name_contiguous(station_profile: dict, bmus: pd.DataFrame) -> (pd.Series, pd.Series):
    lead_party_count = bmus["leadPartyName"].apply(lambda x: contiguous_words(station_profile["rego_station_name"], x))
    bmu_count = bmus["bmUnitName"].apply(lambda x: contiguous_words(station_profile["rego_station_name"], x))
    # max_count = pd.Series(map(max, lead_party_count, bmu_count))
    max_count = pd.Series([max(x, y) for x, y in zip(lead_party_count, bmu_count)])
    max_count_filter = (max_count > 0) & (max_count == max(max_count))
    return max_count, max_count_filter


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
    max_count = pd.Series([max(x, y) for x, y in zip(lead_party_count, bmu_intersection_count)])
    max_count_filter = (max_count > 0) & (max_count == max(max_count))
    return max_count, max_count_filter


def filter_on_fuel_type(station_profile: dict, bmus: pd.DataFrame) -> pd.Series:
    _, filter = intersection(
        bmus["fuelType"],
        station_profile["rego_station_technology"],
    )
    return filter


def filter_on_generation_capacity(station_profile: dict, bmus: pd.DataFrame) -> pd.Series:
    return (station_profile["rego_station_dnc_mw"] / 10 < bmus["generationCapacity"]) & (
        bmus["generationCapacity"] < station_profile["rego_station_dnc_mw"] * 2
    )


################################################################################
# APPLY FILTERS
################################################################################
def define_bmu_match_features_and_filters(station_profile: dict, bmus: pd.DataFrame) -> (pd.DataFrame, list):
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


def apply_bmu_match_filters(bmus: pd.DataFrame, filters: list) -> pd.DataFrame:
    filtered_bmus = bmus.loc[np.logical_and.reduce(filters)]

    try:
        assert len(filtered_bmus) > 0
    except AssertionError:
        warning = "No matching BMUs found"
        raise MappingException(warning)

    return filtered_bmus
