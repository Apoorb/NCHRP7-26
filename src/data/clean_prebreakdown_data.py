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
path_site_char = os.path.join(path_meta, "NCHRP 07-26_Master_Database_shared.xlsx")
path_prebreakdown_all = os.path.join(path_to_project, "data", "interim",
                                     "pre_breakdown_data", "*.csv")
path_prebreakdown_all = glob.glob(path_prebreakdown_all)
path_figures = os.path.join(path_to_project,
                                     r"/figures")


def read_site_data(path_, sheet_name_):
    site_sum = pd.read_excel(path_site_char, sheet_name=sheet_name_, skiprows=1)
    site_sum.columns = [
        inflection.underscore(re.sub("\W+", "_", col.strip()))
        for col in site_sum.columns
    ]
    site_sum.columns = [re.sub("_$", "", col) for col in site_sum.columns]
    return site_sum


def clean_site_sum_merge(site_sum_merge_):
    site_sum_merge_fil_ = site_sum_merge_.assign(
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
        ramp_metering=lambda df: df.ramp_metering.fillna("no").str.lower(),
        fix_ffs=lambda df: np.select(
            [df.fix_ffs.isna(), ~df.fix_ffs.isna()], [False, True]
        ),
        breakdowns_by_tot=lambda df: df.breakdown_events / df.total_counts,
    ).filter(
        items=[
            "site_id",
            "file_name",
            "file_no",
            "file",
            "ffs",
            "fix_ffs",
            "total_counts",
            "breakdown_events",
            "estimated_capacity_veh_hr_ln",
            "number_of_mainline_lane_downstream",
            "number_of_mainline_lane_upstream",
            "number_of_on_ramp_lanes_at_ramp_terminal",
            "hv",
            "mainline_grade",
            "ramp_metering",
            "length_of_acceleration_lane",
            "mainline_aadt_2018",
            "breakdowns_by_tot",
        ],
        axis=1,
    )
    return site_sum_merge_fil_

def get_geometry_list(path_prebreakdown_all_):
    re_geometry_type = re.compile(
        "^[0-9]{1,3}_(?P<geometry_type>.*)_[0-9]{1,2}_?\w*?_pre_brkdn.csv$"
    )
    return ([re.search(re_geometry_type, path.split("\\")[-1]).group("geometry_type")
             for path in path_prebreakdown_all_])



def get_prebreakdown_data(path_prebreakdown_all_, clean_geometry_type_list_=[]):
    prebreakdown_df_list = []
    for path in path_prebreakdown_all_:
        file = path.split("\\")[-1]
        re_site_name = re.compile(
            "^(?P<site_name>[0-9]{1,3}_.*_[0-9]{1,2}_?\w*?)_pre_brkdn.csv$"
        )
        re_site_sno = re.compile(
            "^(?P<site_sno>[0-9]{1,3})_.*_[0-9]{1,2}_?\w*?_pre_brkdn.csv$"
        )
        re_geometry_type = re.compile(
            "^[0-9]{1,3}_(?P<geometry_type>.*)_[0-9]{1,2}_?\w*?_pre_brkdn.csv$"
        )
        site_name = re.search(re_site_name, file).group("site_name")
        site_sno = int(re.search(re_site_sno, file).group("site_sno"))
        geometry_type = re.search(re_geometry_type, file).group("geometry_type")
        if geometry_type not in clean_geometry_type_list_:
            continue

        prebreakdown_df_list.append(
            pd.read_csv(path)
                .assign(file_name=site_name, file_no=site_sno)
                .rename(
                columns={
                    "MainlineVol": "prebreakdown_vol",
                    "MainlineSpeed": "prebreakdown_speed",
                }
            )
        )
    prebreakdown_df_ = pd.concat(prebreakdown_df_list)
    return prebreakdown_df_


if __name__ == "__main__":
    site_sum_merge = read_site_data(path_=path_site_char, sheet_name_="Merge")
    # site_sum_diverge = read_site_data(path_=path_site_char, sheet_name_="Diverge")
    # site_sum_weave = read_site_data(path_=path_site_char, sheet_name_="Weaving")

    site_sum_merge_fil = clean_site_sum_merge(site_sum_merge)
    get_geometry_list(path_prebreakdown_all)
    clean_geometry_type_list = ["Simple Merge", "Ramp Metered"]
    prebreakdown_df_merge = get_prebreakdown_data(
        path_prebreakdown_all_=path_prebreakdown_all,
        clean_geometry_type_list_=clean_geometry_type_list
    )
    clean_geometry_type_list = ["Simple Diverge"]
    prebreakdown_df_diverge = get_prebreakdown_data(
        path_prebreakdown_all_=path_prebreakdown_all,
        clean_geometry_type_list_=clean_geometry_type_list
    )
