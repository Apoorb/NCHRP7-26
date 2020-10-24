import os
import pandas as pd
import numpy as np
import inflection
import re
from sklearn import tree
from sklearn import preprocessing
import graphviz
import seaborn as sns

path_meta = (
    r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - "
    r"NCHRP 07-26\Task 6 - Execute Phase II Plan\Site Analysis & Metadata"
)
path_site_char = os.path.join(path_meta, "NCHRP 07-26_Master_Database_shared.xlsx")
path_prebreakdown = os.path.join(
    r"C:\Users\abibeka\Documents_axb\nchrp7-26\regression_analysis_prebreakdown",
    "pre_breakdown_analysis_v1.xlsx",
)
path_figures = r"/figures"
site_char = pd.read_excel(path_site_char, sheet_name="Merge", skiprows=1)
site_char.columns = [
    inflection.underscore(re.sub("\W+", "_", col.strip())) for col in site_char.columns
]
site_char.columns = [re.sub("_$", "", col) for col in site_char.columns]
site_char.columns
site_char.number_of_mainline_lane_downstream


site_char_fil = site_char.assign(
    file_name=lambda df: df.file_name.str.strip(),
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
    length_of_acceleration_lane=lambda df: df.length_of_acceleration_lane.astype(str)
    .str.extract(r"(\d+)")
    .astype(int),
    ramp_metering=lambda df: df.ramp_metering.fillna("No"),
    fix_ffs=lambda df: np.select(
        [df.fix_ffs.isna(), ~df.fix_ffs.isna()], [False, True]
    ),
).filter(
    items=[
        "site_id",
        "file_name",
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
    ],
    axis=1,
)


x1 = pd.ExcelFile(path_prebreakdown)
x1.sheet_names
list_prebreakdown_df = []
for sheet_nm in x1.sheet_names:
    if sheet_nm == "pivot_tables":
        continue
    temp_df = x1.parse(sheet_nm).assign(file=sheet_nm.strip())
    temp_df = temp_df.rename(
        columns={col: inflection.underscore(col) for col in temp_df.columns}
    )
    list_prebreakdown_df.append(temp_df)

prebreakdown_df = pd.concat(list_prebreakdown_df)

prebreakdown_df_meta = prebreakdown_df.merge(site_char_fil, on="file", how="left")


prebreakdown_df_meta = (
    prebreakdown_df_meta.filter(
        items=[
            # "site_id",
            "ffs",
            # "fix_ffs",
            "total_counts",
            "breakdown_events",
            "estimated_capacity_veh_hr_ln",
            "number_of_mainline_lane_upstream",
            "number_of_on_ramp_lanes_at_ramp_terminal",
            "hv",
            "mainline_grade",
            "ramp_metering",
            "length_of_acceleration_lane",
            "mainline_speed",
            "ramp_vol",
            "mainline_vol",
            "mainline_aadt_2018",
        ],
        axis=1,
    )
    .rename(columns={"mainline_vol": "prebreakdown_volume"})
    .assign(breakdowns_by_tot=lambda df: df.breakdown_events / df.total_counts)
)

sns.pairplot(
    prebreakdown_df_meta[
        [
            "prebreakdown_volume",
            "ramp_vol",
            "number_of_mainline_lane_upstream",
            "length_of_acceleration_lane",
            "mainline_aadt_2018",
            "breakdowns_by_tot",
        ]
    ]
)

lb = preprocessing.LabelBinarizer()
prebreakdown_df_meta.ramp_metering = lb.fit_transform(
    prebreakdown_df_meta.ramp_metering
)
lb.classes_
lb.get_params()
lb.inverse_transform(prebreakdown_df_meta.ramp_metering)

y = prebreakdown_df_meta.mainline_vol
X = prebreakdown_df_meta[
    [
        col
        for col in prebreakdown_df_meta.columns
        if col not in ["mainline_vol", "estimated_capacity_veh_hr_ln"]
    ]
]
max_depth_ = 6
clf = tree.DecisionTreeRegressor(max_depth=max_depth_)
clf = clf.fit(X, y)


dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=[y.name],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render(os.path.join(path_figures, f"tree_depth_{max_depth_}"))
