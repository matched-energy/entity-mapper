import copy
from pathlib import Path

import click
import pandas as pd

import scores.analytics.rego_analysis
import scores.core.supplier_gen_by_tech_by_month


def add_columns(regos):
    _regos = copy.deepcopy(regos)
    _regos["GWh"] = _regos["MWh Per Certificate"] * _regos["No. Of Certificates"] / 1e3
    _regos["tech_simple"] = regos["Technology Group"].map(
        scores.analytics.rego_analysis.tech_simple
    )
    return _regos


def filter(regos):
    return regos[
        (regos["Certificate Status"] == "Redeemed") & (regos["Scheme"] == "REGO")
    ]


def groupby_status(regos):
    regos_by_status = (
        regos.groupby(["Certificate Status"]).agg(GWh=("GWh", "sum"))
        # .sort_values(by="MWh", ascending=False)
    )
    regos_by_status["%"] = regos_by_status["GWh"] / regos_by_status["GWh"].sum() * 100
    return regos_by_status


def groupby_tech(regos):
    regos_by_tech = (
        regos.groupby(["tech_simple", "Technology Group"])
        .agg(GWh=("GWh", "sum"))
        .sort_values(by="GWh", ascending=False)
    )
    regos_by_tech["%"] = regos_by_tech["GWh"] / regos_by_tech["GWh"].sum() * 100
    return regos_by_tech


def groupby_station(regos):
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
    regos_by_station["%"] = (
        regos_by_station["GWh"] / regos_by_station["GWh"].sum() * 100
    )
    return regos_by_station.reset_index()


def hist_by_volume(groupby_station):
    _groupby_station = copy.deepcopy(groupby_station)
    bins = [-float("inf"), 0, 1, 1e1, 1e2, 1e3, 1e4, 1e5, float("inf")]
    _groupby_station["cuts"] = pd.cut(
        groupby_station["GWh"],
        bins=bins,
        labels=bins[1:],
        right=True,
    )

    hist = pd.DataFrame(_groupby_station["cuts"].value_counts())
    hist["average_MW"] = [
        f"{float(vol) * 1000 / (365 * 24):,.1f}" for vol in hist.index
    ]
    hist["GWh"] = _groupby_station.groupby("cuts").agg(GWh=("GWh", "sum"))
    hist["% count"] = hist["count"] / hist["count"].sum() * 100
    hist["% GWh"] = hist["GWh"] / hist["GWh"].sum() * 100

    return hist[["average_MW", "count", "% count", "GWh", "% GWh"]]


def stats_by_station(regos, station):
    _regos = regos[regos["Generating Station / Agent Group"] == station]
    return _regos


def main(regos_path: Path):
    regos = scores.core.supplier_gen_by_tech_by_month.read(regos_path)
    regos = scores.core.supplier_gen_by_tech_by_month.parse_output_period(regos)
    regos = add_columns(regos)
    regos = filter(regos)
    return regos


@click.command()
@click.option(
    "--regos-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to REGO csv",
)
def main_click(regos_path: Path):
    main(regos_path)


if __name__ == "__main__":
    main_click()
