import os
import pandas as pd
import numpy as np
import inflection
import re
import glob
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"
from sklearn import tree
from sklearn import preprocessing
import graphviz
import seaborn as sns

path_meta = (
    r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - "
    r"NCHRP 07-26\Task 6 - Execute Phase II Plan\Site Analysis & Metadata"
)
path_site_char = os.path.join(path_meta, "NCHRP 07-26_Master_Database_shared.xlsx")
path_prebreakdown_all = os.path.join(r"/data/interim/pre_breakdown_data", "*.csv",)
path_prebreakdown_all = glob.glob(path_prebreakdown_all)
path_figures = r"/figures"
site_sum = pd.read_excel(path_site_char, sheet_name="Merge", skiprows=1)
site_sum.columns = [
    inflection.underscore(re.sub("\W+", "_", col.strip())) for col in site_sum.columns
]
site_sum.columns = [re.sub("_$", "", col) for col in site_sum.columns]


site_sum_fil = site_sum.assign(
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
    length_of_acceleration_lane=lambda df: df.length_of_acceleration_lane.astype(str)
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

prebreakdown_df_list = []
for path in path_prebreakdown_all:
    if path.find("Simple Merge") == -1 & path.find("Ramp Metered") == -1:
        continue
    file = path.split("\\")[-1]
    re_geometry_name = re.compile(
        "^(?P<geometry_type>[0-9]{1,3}_.*_[0-9]{1,2}_?\w*?)_pre_brkdn.csv$"
    )
    re_geometry_sno = re.compile(
        "^(?P<site_sno>[0-9]{1,3})_.*_[0-9]{1,2}_?\w*?_pre_brkdn.csv$"
    )
    geometry_name = re.search(re_geometry_name, file).group("geometry_type")
    geometry_sno = int(re.search(re_geometry_sno, file).group("site_sno"))
    prebreakdown_df_list.append(
        pd.read_csv(path)
        .assign(file_name=geometry_name, file_no=geometry_sno)
        .rename(
            columns={
                "MainlineVol": "prebreakdown_vol",
                "MainlineSpeed": "prebreakdown_speed",
            }
        )
    )
prebreakdown_df = pd.concat(prebreakdown_df_list)

prebreakdown_df_meta = prebreakdown_df.merge(site_sum_fil, on="file_no", how="left")
prebreakdown_df_meta.rename(columns={"file_name_x": "file_name"}, inplace=True)
site_no_name_dict = {
    int(site.split("_")[0]): site for site in prebreakdown_df_meta.file_name.unique()
}
site_no_name_dict = dict(sorted(site_no_name_dict.items()))
site_name_sorted = list(site_no_name_dict.values())

fig = px.box(
    prebreakdown_df_meta,
    x="file_name",
    y="prebreakdown_vol",
    color="ramp_metering",
    hover_data=[
        "ffs",
        "fix_ffs",
        "number_of_mainline_lane_downstream",
        "length_of_acceleration_lane",
        "mainline_aadt_2018",
        "estimated_capacity_veh_hr_ln",
    ],
)
fig.update_layout(
    font=dict(family="Arial", size=18,),
    xaxis={"categoryorder": "array", "categoryarray": site_name_sorted},
    hoverlabel=dict(bgcolor="white", font_size=18, font_family="Rockwell"),
)
fig.update_layout()
fig.show()
fig.write_html(os.path.join(path_figures, "box_plots_simple_merge.html"))


fig2 = px.scatter_matrix(
    prebreakdown_df_meta,
    dimensions=[
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_acceleration_lane",
        "mainline_aadt_2018",
        "breakdowns_by_tot",
        "ffs",
    ],
    labels={
        "prebreakdown_vol": "prebreakdown_vol",
        "prebreakdown_speed": "prebreakdown_speed",
        "length_of_acceleration_lane": "len acc lane",
        "mainline_aadt_2018": "aadt_2018",
        "breakdowns_by_tot": "breakdown/total_count",
        "ffs": "ffs",
    },
)
fig2.show()
fig2.write_html(os.path.join(path_figures, "pair_plots_simple_merge.html"))


fig3 = px.scatter(
    prebreakdown_df_meta,
    x="mainline_aadt_2018",
    y="prebreakdown_speed",
    color="file_name",
)
fig3.show()
fig3.write_html(os.path.join(path_figures, "pair_plots_simple_merge.html"))


prebreakdown_df_meta_simple_merge = prebreakdown_df_meta.query("file_no <=26")

prebreakdown_df_meta_simple_merge = prebreakdown_df_meta_simple_merge.filter(
    items=[
        # "site_id",
        "ffs",
        # "fix_ffs",
        "total_counts",
        "breakdown_events",
        "estimated_capacity_veh_hr_ln",
        "number_of_mainline_lane_upstream",
        "number_of_on_ramp_lanes_at_ramp_terminal",
        "ramp_metering",
        "length_of_acceleration_lane",
        "prebreakdown_speed",
        "ramp_vol",
        "prebreakdown_vol",
        "mainline_aadt_2018",
    ],
    axis=1,
)
# fig = px.scatter(
#     prebreakdown_df_meta_simple_merge,
#     x="mainline_aadt_2018",
#     y="prebreakdown_vol",
#     color="file_name"
# )
# fig.show()


lb = preprocessing.LabelBinarizer()
prebreakdown_df_meta_simple_merge.ramp_metering = lb.fit_transform(
    prebreakdown_df_meta_simple_merge.ramp_metering
)
lb.classes_
lb.get_params()
lb.inverse_transform(prebreakdown_df_meta_simple_merge.ramp_metering)

y = prebreakdown_df_meta_simple_merge.prebreakdown_vol
X = prebreakdown_df_meta_simple_merge[
    [
        col
        for col in prebreakdown_df_meta_simple_merge.columns
        if col not in ["prebreakdown_vol", "estimated_capacity_veh_hr_ln"]
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
graph.render(os.path.join(path_figures, f"all_simple_merge_tree_depth_{max_depth_}"))
