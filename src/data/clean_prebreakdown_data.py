import os
import pandas as pd
import numpy as np
import inflection
import re
import glob
from src.utils import get_project_root

# Set paths:
path_to_project = str(get_project_root())
path_meta = (
    r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - "
    r"NCHRP 07-26\Task 6 - Execute Phase II Plan\Site Analysis & Metadata"
)
path_capacity_df = os.path.join(
    r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - NCHRP 07-26"
    r"\Task 6 - Execute Phase II Plan\Sample Data Analysis\Updated Analysis",
    "Data_output-Final.csv",
)
path_site_char = os.path.join(path_meta, "NCHRP 07-26_Master_Database_shared.xlsx")
path_prebreakdown_all = os.path.join(
    path_to_project, "data", "interim", "pre_breakdown_data", "*.csv"
)
path_prebreakdown_all = glob.glob(path_prebreakdown_all)
path_interim = os.path.join(path_to_project, "data", "interim")


def read_site_data(path_, sheet_name_, nrows_, usecols_):
    site_sum = pd.read_excel(
        path_site_char,
        sheet_name=sheet_name_,
        skiprows=1,
        nrows=nrows_,
        usecols=usecols_,
    )
    site_sum.columns = [
        inflection.underscore(re.sub("\W+", "_", col.strip()))
        for col in site_sum.columns
    ]
    site_sum.columns = [re.sub("_$", "", col) for col in site_sum.columns]
    return site_sum


def clean_site_sum_merge(site_sum_merge_, iloc_max_row):
    site_sum_merge_fil_ = (
        site_sum_merge_.iloc[0:iloc_max_row, :]
        .assign(
            file_name=lambda df: df.file_name.str.strip(),
            file_no=lambda df: df.file_name.str.split("_", expand=True)[0].astype(int),
            number_of_mainline_lane_upstream=lambda df: df.number_of_mainline_lane_upstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            number_of_mainline_lane_downstream=lambda df: df.number_of_mainline_lane_downstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            number_of_on_ramp_lanes_at_ramp_terminal=lambda df: df.number_of_on_ramp_lanes_at_ramp_terminal.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            length_of_acceleration_lane=lambda df: df.length_of_acceleration_lane.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            ramp_metering=lambda df: np.select(
                [
                    df.metered_ramp.str.upper() == "NO",
                    df.metered_ramp.str.upper() == "YES",
                ],
                [False, True],
            ),
            fix_ffs=lambda df: np.select(
                [df.fix_ffs.isna(), ~df.fix_ffs.isna()], [False, True]
            ),
            presence_of_adjacent_ramps=lambda df: np.select(
                [
                    df.presence_of_adjacent_ramps.str.upper() == "NO",
                    df.presence_of_adjacent_ramps.str.upper() == "YES",
                ],
                [False, True],
            ),
            upstream_ramp_type_on_off=lambda df: df.upstream_ramp_type_on_off.str.lower(),
            downstream_ramp_type_on_off=lambda df: df.downstream_ramp_type_on_off.str.lower(),
            fwy_to_fwy_ramp=lambda df: df.fwy_to_fwy_ramp.str.lower(),
            signal_ramp_terminal=lambda df: df.signal_ramp_terminal.str.lower(),
            free_flow_ramp_terminal=lambda df: df.free_flow_ramp_terminal.str.lower(),
            roundabout_terminal=lambda df: df.roundabout_terminal.str.lower(),
            area_type=lambda df: df.area_type.str.lower(),
        )
        .filter(
            items=[
                "site_id",
                "file_name",
                "file_no",
                "file",
                "lat",
                "long",
                "number_of_mainline_lane_downstream",
                "number_of_mainline_lane_upstream",
                "number_of_on_ramp_lanes_at_ramp_terminal",
                "number_of_on_ramp_lane_gore",
                "presence_of_adjacent_ramps",
                "lane_drop_y_n" "hv",
                "mainline_grade",
                "ramp_metering",
                "length_of_acceleration_lane",
                "mainline_speed_limit",
                "mainline_aadt",
                "area_type",
                "dist_to_upstream_ramp_ft",
                "upstream_ramp_type_on_off",
                "dist_to_downstream_ramp_ft",
                "downstream_ramp_type_on_off",
                "fwy_to_fwy_ramp",
                "signal_ramp_terminal",
                "free_flow_ramp_terminal",
                "roundabout_terminal",
            ],
            axis=1,
        )
    )
    for col in site_sum_merge_fil_:
        print(site_sum_merge_fil_[col].unique())
    return site_sum_merge_fil_


def clean_site_sum_diverge(site_sum_diverge_, iloc_max_row):
    site_sum_diverge_fil_ = (
        site_sum_diverge_.iloc[0:iloc_max_row, :]
        .assign(
            file_name=lambda df: df.file_name.str.strip(),
            file_no=lambda df: df.file_name.str.split("_", expand=True)[0].astype(int),
            number_of_mainline_lane_upstream=lambda df: df.number_of_mainline_lane_upstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            number_of_mainline_lane_downstream=lambda df: df.number_of_mainline_lane_downstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            number_of_off_ramp_lane=lambda df: df.number_of_off_ramp_lane.astype(str)
            .str.extract(r"(\d+)")
            .astype(int),
            length_of_deceleration_lane=lambda df: df.length_of_deceleration_lane.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(int),
            presence_of_adjacent_ramps=lambda df: np.select(
                [
                    df.presence_of_adjacent_ramps.str.upper() == "NO",
                    df.presence_of_adjacent_ramps.str.upper() == "YES",
                ],
                [False, True],
            ),
            upstream_ramp_type_on_off=lambda df: df.upstream_ramp_type_on_off.str.lower(),
            downstream_ramp_type_on_off=lambda df: df.downstream_ramp_type_on_off.str.lower(),
            fwy_to_fwy_ramp=lambda df: df.fwy_to_fwy_ramp.str.lower(),
            signal_ramp_terminal=lambda df: df.signal_ramp_terminal.str.lower(),
            free_flow_ramp_terminal=lambda df: df.free_flow_ramp_terminal.str.lower(),
            roundabout_terminal=lambda df: df.roundabout_terminal.str.lower(),
            stop_sign_terminal=lambda df: df.stop_sign_terminal.str.lower(),
            area_type=lambda df: df.area_type.str.lower(),
        )
        .filter(
            items=[
                "site_id",
                "file_name",
                "file_no",
                "file",
                "lat",
                "long",
                "number_of_mainline_lane_downstream",
                "number_of_mainline_lane_upstream",
                "number_of_off_ramp_lane",
                "presence_of_adjacent_ramps",
                "hv",
                "mainline_grade",
                "length_of_deceleration_lane",
                "mainline_aadt",
                "mainline_speed_limit",
                "dist_to_upstream_ramp_ft",
                "upstream_ramp_type_on_off",
                "dist_to_downstream_ramp_ft",
                "downstream_ramp_type_on_off",
                "fwy_to_fwy_ramp",
                "signal_ramp_terminal",
                "free_flow_ramp_terminal",
                "roundabout_terminal",
                "area_type",
            ],
            axis=1,
        )
    )
    for col in site_sum_diverge_fil_:
        print(site_sum_diverge_fil_[col].unique())
    return site_sum_diverge_fil_


def clean_site_sum_weave(site_sum_weave_, iloc_max_row):
    site_sum_weave_fil_ = (
        site_sum_weave_.iloc[0:iloc_max_row, :]
        .assign(
            file_name=lambda df: df.file_name.str.strip(),
            file_no=lambda df: df.file_name.str.split("_", expand=True)[0].astype(int),
            lat=lambda df: df.location.str.split(",", expand=True)[0],
            long=lambda df: df.location.str.split(",", expand=True)[1],
            num_of_mainline_lane_weaving_section=lambda df: df.num_of_mainline_lane_weaving_section.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(float),
            number_of_mainline_lane_upstream=lambda df: df.number_of_mainline_lane_upstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(float),
            number_of_mainline_lane_downstream=lambda df: df.number_of_mainline_lane_downstream.astype(
                str
            )
            .str.extract(r"(\d+)")
            .astype(float),
            fix_ffs=lambda df: np.select(
                [df.ffs_fixed.isna(), ~df.ffs_fixed.isna()], [False, True]
            ),
            ramp_metering=lambda df: np.select(
                [
                    df.metered_ramp.str.upper() == "NO",
                    df.metered_ramp.str.upper() == "YES",
                ],
                [False, True],
            ),
            upstream_ramp_type_on_off=lambda df: df.upstream_ramp_type_on_off.str.lower(),
            downstream_ramp_type_on_off=lambda df: df.downstream_ramp_type_on_off.str.lower(),
            fwy_to_fwy_ramp_on=lambda df: df.fwy_to_fwy_ramp.str.lower(),
            signal_ramp_terminal_on=lambda df: df.signal_ramp_terminal.str.lower(),
            free_flow_ramp_terminal_on=lambda df: df.free_flow_ramp_terminal.str.lower(),
            roundabout_terminal_on=lambda df: df.roundabout_terminal.str.lower(),
            fwy_to_fwy_ramp_off=lambda df: df.fwy_to_fwy_ramp.str.lower(),
            signal_ramp_terminal_off=lambda df: df.signal_ramp_terminal.str.lower(),
            free_flow_ramp_terminal_off=lambda df: df.free_flow_ramp_terminal.str.lower(),
            roundabout_terminal_off=lambda df: df.roundabout_terminal.str.lower(),
            area_type=lambda df: df.area_type.str.lower(),
        )
        .filter(
            items=[
                "site_id",
                "file_name",
                "file_no",
                "file",
                "lat",
                "long",
                "lcrf",
                "lcfr",
                "lcrr",
                "n_wl",
                "mainline_speed_limit",
                "short_length_ls_ft",
                "base_length_lb_ft",
                "number_of_mainline_lane_downstream",
                "number_of_mainline_lane_upstream",
                "num_of_mainline_lane_weaving_section",
                "interchange_density",
                "hv",
                "mainline_grade",
                "ramp_metering",
                "length_of_deceleration_lane",
                "mainline_aadt",
                "fwy_to_fwy_ramp_on",
                "signal_ramp_terminal_on",
                "free_flow_ramp_terminal_on",
                "roundabout_terminal_on",
                "fwy_to_fwy_ramp_off",
                "signal_ramp_terminal_off",
                "free_flow_ramp_terminal_off",
                "roundabout_terminal_off",
                "area_type",
            ],
            axis=1,
        )
    )
    for col in site_sum_weave_fil_:
        print(site_sum_weave_fil_[col].unique())
    return site_sum_weave_fil_


def get_geometry_list(path_prebreakdown_all_):
    re_geometry_type = re.compile(
        "^[0-9]{1,3}_(?P<geometry_type>.*)_[0-9]{1,2}_?\w*?_pre_brkdn.csv$"
    )
    return set(
        [
            re.search(re_geometry_type, path.split("\\")[-1])
            .group("geometry_type")
            .capitalize()
            for path in path_prebreakdown_all_
            if "_pre_brkdn.csv" in path
        ]
    )


def get_prebreakdown_data(path_prebreakdown_all_, clean_geometry_type_list_=[]):
    prebreakdown_df_list = []
    for path in path_prebreakdown_all_:
        file = path.split("\\")[-1]
        re_pat = re.compile(
            "^(?P<site_name>(?P<site_sno>[0-9]{1,3})_(?P<geometry_type>.*)_[0-9]{1,2}_?\w*?)(?P<data_type>(_pre_brkdn.csv$)|(_uncongested.csv$))"
        )
        site_name = re.search(re_pat, file).group("site_name")
        data_type = re.search(re_pat, file).group("data_type")
        site_sno = int(re.search(re_pat, file).group("site_sno"))
        geometry_type = re.search(re_pat, file).group("geometry_type").capitalize()
        if data_type == "_pre_brkdn.csv":
            failure = 1
        else:
            failure = 0
        if geometry_type not in clean_geometry_type_list_:
            continue
        prebreakdown_df_list.append(
            pd.read_csv(path)
            .assign(
                file_name=site_name,
                file_no=site_sno,
                geometry_type=geometry_type,
                failure=failure,
            )
            .rename(
                columns={
                    "MainlineVol": "mainline_vol",
                    "MainlineSpeed": "mainline_speed",
                }
            )
        )
    prebreakdown_df_ = pd.concat(prebreakdown_df_list)
    return prebreakdown_df_


if __name__ == "__main__":
    cap_df = pd.read_csv(path_capacity_df)
    cap_df.columns = [
        inflection.underscore(re.sub("\W+", "", col)) for col in cap_df.columns
    ]
    cap_df_mod = (
        cap_df.filter(
            items=[
                "sl_no",
                "file",
                "total_counts",
                "ffs",
                "breakdown_events",
                "alpha",
                "beta",
                "estimated_capacity",
            ]
        )
        .rename(
            columns={
                "ffs": "ffs_cap_df",
                "sl_no": "file_no",
                "file": "file_name_cap_df",
            }
        )
        .assign(file_no=lambda df: df.file_no.astype(int))
    )
    site_sum_merge = read_site_data(
        path_=path_site_char, sheet_name_="Merge", nrows_=41, usecols_=range(0, 60)
    )
    site_sum_merge_fil = clean_site_sum_merge(site_sum_merge, iloc_max_row=None)

    site_sum_merge_fil = site_sum_merge_fil.merge(cap_df_mod, on="file_no", how="left")
    site_sum_merge_fil.to_csv(
        os.path.join(path_interim, "all_merge_meta.csv"), index=False
    )
    get_geometry_list(path_prebreakdown_all)
    clean_geometry_type_list = ["Simple merge", "Ramp metered"]
    prebreakdown_df_simple_merge = get_prebreakdown_data(
        path_prebreakdown_all_=path_prebreakdown_all,
        clean_geometry_type_list_=clean_geometry_type_list,
    )
    clean_geometry_type_list = get_geometry_list(path_prebreakdown_all)
    prebreakdown_df_all = get_prebreakdown_data(
        path_prebreakdown_all_=path_prebreakdown_all,
        clean_geometry_type_list_=clean_geometry_type_list,
    )
    prebreakdown_df_simple_merge_meta = (
        prebreakdown_df_simple_merge.merge(site_sum_merge_fil, on="file_no", how="left")
        .rename(columns={"file_name_x": "file_name"})
        .drop(columns="file_name_y")
    )
    prebreakdown_df_merge_all_meta = (
        prebreakdown_df_all.merge(site_sum_merge_fil, on="file_no", how="right")
        .rename(columns={"file_name_x": "file_name"})
        .drop(columns="file_name_y")
    )
    prebreakdown_df_merge_all_meta.to_csv(
        os.path.join(path_interim, "prebreakdown_df_all_merge_meta.csv"), index=False
    )
    prebreakdown_df_simple_merge_meta.to_csv(
        os.path.join(path_interim, "prebreakdown_simple_merge_and_meta.csv"),
        index=False,
    )

    site_sum_diverge = read_site_data(
        path_=path_site_char, sheet_name_="Diverge", nrows_=42, usecols_=range(0, 61)
    )
    site_sum_diverge_fil = clean_site_sum_diverge(site_sum_diverge, iloc_max_row=None)
    site_sum_diverge_fil = site_sum_diverge_fil.merge(
        cap_df_mod, on="file_no", how="left"
    )

    site_sum_weave = read_site_data(
        path_=path_site_char, sheet_name_="Weaving", nrows_=26, usecols_=range(0, 81)
    )
    site_sum_weave_fil = clean_site_sum_weave(site_sum_weave, iloc_max_row=26)
    site_sum_weave_fil = site_sum_weave_fil.merge(cap_df_mod, on="file_no", how="left")
    site_sum_diverge_fil.to_csv(
        os.path.join(path_interim, "all_diverge_meta.csv"), index=False
    )
    site_sum_weave_fil.to_csv(
        os.path.join(path_interim, "all_weave_meta.csv"), index=False
    )
    prebreakdown_df_diverge_meta = (
        prebreakdown_df_all.merge(site_sum_diverge_fil, on="file_no", how="right")
        .rename(columns={"file_name_x": "file_name"})
        .drop(columns="file_name_y")
    )
    len(prebreakdown_df_diverge_meta.file_name.unique())
    prebreakdown_df_weave_meta = (
        prebreakdown_df_all.merge(site_sum_weave_fil, on="file_no", how="right")
        .rename(columns={"file_name_x": "file_name"})
        .drop(columns="file_name_y")
    )
    len(prebreakdown_df_weave_meta.file_name.unique())
    prebreakdown_df_diverge_meta.to_csv(
        os.path.join(path_interim, "cap_diverge_df.csv"), index=False
    )
    prebreakdown_df_weave_meta.to_csv(
        os.path.join(path_interim, "cap_weave_df.csv"), index=False
    )
