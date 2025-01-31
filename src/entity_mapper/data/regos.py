import copy
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from entity_mapper.common import MappingException

COLUMNS: list[str] = [
    "Accreditation No.",
    "Generating Station / Agent Group",
    "Station TIC",
    "Scheme",
    "Country",
    "Technology Group",
    "Generation Type",
    "Output Period",
    "No. Of Certificates",
    "Start Certificate No.",
    "End Certificate No.",
    "MWh Per Certificate",
    "Issue Date",
    "Certificate Status",
    "Status Date",
    "Current Holder Organisation Name",
    "Company Registration Number",
]

TECH_SIMPLE = {
    "Photovoltaic": "SOLAR",
    "Hydro": "HYDRO",
    "Wind": "WIND",
    "Biomass": "BIOMASS",
    "Biogas": "BIOMASS",
    "Landfill Gas": "BIOMASS",
    "On-shore Wind": "WIND",
    "Hydro 20MW DNC or less": "HYDRO",
    "Fuelled": "BIOMASS",
    "Off-shore Wind": "WIND",
    "Micro Hydro": "HYDRO",
    "Biomass 50kW DNC or less": "BIOMASS",
}


def read_from_file(
    filepath: Path,
    current_holder_organisation_names: Optional[list[str]] = None,
    status: str = "Redeemed",
) -> pd.DataFrame:
    d = pd.read_csv(filepath, names=COLUMNS, skiprows=4)
    if current_holder_organisation_names:
        return d[
            (d["Current Holder Organisation Name"].isin(current_holder_organisation_names))
            & (d["Certificate Status"] == status)
            ## TODO: this is a bug! Should do this filtering in the 'filter function'?
        ]
    else:
        return d


def parse_output_period(df_regos: pd.DataFrame) -> pd.DataFrame:
    df = df_regos.copy(deep=True)
    if df.empty:
        df[["start", "end", "months_different"]] = pd.DataFrame(columns=["start", "end", "months_difference"])
        return df

    # TODO: test
    def parse_date_range(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        if "/" in date_str:
            start, end = date_str.split(" - ")
            return (
                pd.to_datetime(start, dayfirst=True),
                pd.to_datetime(end, dayfirst=True),
            )
        elif " - " in date_str:
            year_start, year_end = date_str.split(" - ")
            start_dt = pd.to_datetime("01/01/" + year_start, dayfirst=True)
            end_dt = pd.to_datetime("31/12/" + year_end, dayfirst=True)
            return start_dt, end_dt
        elif "-" in date_str:
            month_year = pd.to_datetime(date_str, format="%b-%Y")
            start_dt = month_year.replace(day=1)
            end_dt = month_year + pd.offsets.MonthEnd(0)
            return start_dt, end_dt
        else:
            raise ValueError(r"Invalid date string {}".format(date_str))

    df[["start", "end"]] = df["Output Period"].apply(lambda x: pd.Series(parse_date_range(x)))
    df["end"] += np.timedelta64(1, "D")
    df["months_difference"] = df.apply(
        lambda row: relativedelta(row["end"], row["start"]).years * 12 + relativedelta(row["end"], row["start"]).months,
        axis=1,
    )

    # # Add columns for each month to represent if it's within the start/end range
    # for month in range(1, 13):
    #     month_name = f"month_{month}_in_range"
    #     df[month_name] = (df["start"].dt.month <= month) & (df["end"].dt.month >= month)

    # df.to_csv("test.csv")
    return df


def add_columns(regos: pd.DataFrame) -> pd.DataFrame:
    _regos = copy.deepcopy(regos)
    _regos["GWh"] = _regos["MWh Per Certificate"] * _regos["No. Of Certificates"] / 1e3
    _regos["tech_simple"] = regos["Technology Group"].map(TECH_SIMPLE)
    return _regos


def filter(regos: pd.DataFrame) -> pd.DataFrame:
    return regos[(regos["Certificate Status"] == "Redeemed") & (regos["Scheme"] == "REGO")]


def load_regos(regos_path: Path) -> pd.DataFrame:
    regos = read_from_file(regos_path)
    regos = parse_output_period(regos)
    regos = add_columns(regos)
    regos = filter(regos)
    return regos


def load_accredited_stations(accredited_stations_dir: Path) -> pd.DataFrame:
    names = [
        "AccreditationNumber",
        "Status",
        "GeneratingStation",
        "Scheme",
        "StationDNC",
        "Country",
        "Technology",
        "ContractType",
        "AccreditationDate",
        "CommissionDate",
        "Organisation",
        "OrganisationContactAddress",
        "OrganisationContactFax",
        "GeneratingStationAddress",
    ]
    dfs = []

    with os.scandir(accredited_stations_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".csv"):
                filepath = Path(entry.path)
                try:
                    df = pd.read_csv(filepath, skiprows=1, names=names)
                    df["StationDNC_MW"] = df["StationDNC"].astype(float) / 1e3
                    dfs.append(df)
                except ValueError as e:
                    print(f"Skipping {entry.name}: {e}")

    return pd.concat(dfs)


def groupby_regos_by_station(regos: pd.DataFrame) -> pd.DataFrame:
    regos_by_station = (
        regos.groupby("Generating Station / Agent Group")
        .agg(
            accredition_number=("Accreditation No.", "first"),
            company_registration_number=("Company Registration Number", "first"),
            GWh=("GWh", "sum"),
            technology_group=("Technology Group", "first"),
            generation_type=("Generation Type", "first"),
            tech_simple=("tech_simple", "first"),
        )
        .sort_values(by="GWh", ascending=False)
    )
    regos_by_station["%"] = regos_by_station["GWh"] / regos_by_station["GWh"].sum() * 100
    return regos_by_station.reset_index()


def extract_rego_volume_by_month(
    regos: pd.DataFrame,
    rego_station_name: str,
    rego_station_dnc_mw: float,  # TODO - move this
) -> Tuple[dict, pd.DataFrame]:
    # TODO - validate always monthly
    station_regos = regos[regos["Generating Station / Agent Group"] == rego_station_name]
    station_regos_by_period = station_regos.groupby(["start", "end", "months_difference"]).agg(dict(GWh="sum"))
    rego_total_volume = station_regos_by_period["GWh"].sum()
    return (
        dict(
            rego_total_volume=rego_total_volume,
            rego_capacity_factor=(
                rego_total_volume * 1e3 / (rego_station_dnc_mw * 24 * 365)  # NOTE: assuming 1 year!
            ),
            rego_sampling_months=12,  # NOTE: presumed!
        ),
        station_regos_by_period.reset_index().set_index("start").sort_index(),
    )


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
