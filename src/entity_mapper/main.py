from pathlib import Path
from typing import Optional

import pandas as pd
import scores.common.utils

import entity_mapper.utils
from entity_mapper.common import MappingException
from entity_mapper.data.bmus import get_bmu_list_and_aggregate_properties, load_bmus
from entity_mapper.data.regos import (
    get_generator_profile,
    groupby_regos_by_station,
    load_accredited_stations,
    load_regos,
)
from entity_mapper.filter_on_aggregate_data import appraise_energy_volumes, appraise_rated_power
from entity_mapper.filter_on_bmu_meta_data import get_matching_bmus
from entity_mapper.summarise_and_score import summarise_mapping_and_mapping_strength

LOGGER = entity_mapper.utils.get_logger("entity_mapper")


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
