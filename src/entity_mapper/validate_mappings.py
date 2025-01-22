from pathlib import Path

import pandas as pd
import scores.common.utils


def compare_to_expected(mapping_scores: pd.DataFrame, expected_mappings_file: Path) -> pd.DataFrame:
    """Left join"""
    expected_mappings = scores.common.utils.from_yaml_file(expected_mappings_file)
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
