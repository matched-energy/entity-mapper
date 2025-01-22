import pandas as pd
from pandas.testing import assert_frame_equal

import entity_mapper.main


def test_end_to_end() -> None:
    mappings = entity_mapper.main.main(start=0, stop=3)
    expected_mappings = pd.DataFrame(
        [
            {
                "rego_name": "Drax Power Station (REGO)",
                "bmu_ids": "T_DRAXX-1, T_DRAXX-2, T_DRAXX-3, T_DRAXX-4",
            },
            {"rego_name": "Walney Extension", "bmu_ids": "T_WLNYO-3, T_WLNYO-4"},
            {
                "rego_name": "Triton Knoll Offshore Windfarm",
                "bmu_ids": "T_TKNEW-1, T_TKNWW-1",
            },
        ]
    )
    assert_frame_equal(
        mappings[["rego_name", "bmu_ids"]].reset_index(drop=True),
        expected_mappings.reset_index(drop=True),
    )
