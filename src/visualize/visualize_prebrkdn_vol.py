import os
import pandas as pd
import numpy as np
import inflection
import re
import glob
import plotly.express as px
from sklearn import tree
from sklearn import preprocessing
import graphviz
import seaborn as sns
import plotly.io as pio
from src.utils import get_project_root

pio.renderers.default = "browser"

# Set paths:
path_to_project = str(get_project_root())
path_interim = os.path.join(path_to_project, "data", "interim")
path_processed = os.path.join(path_to_project, "data", "processed")
path_figures = os.path.join(path_to_project, "figures")
path_box_fig = os.path.join(path_figures, "box_plots")
if not os.path.exists(path_box_fig):
    os.mkdir(path_box_fig)
path_prebreakdown_merge_and_meta = os.path.join(
    path_interim, "prebreakdown_merge_and_meta.csv"
)
path_prebreakdown_df_all = os.path.join(path_interim, "prebreakdown_merge_no_meta.csv")
path_prebreakdown_df_ufl_meta = os.path.join(path_interim,
                                             "prebreakdown_ufl_merge_and_meta.csv")


def get_correct_sort_order_site_name(prebreakdown_df):
    site_no_name_dict_ = {
        int(site.split("_")[0]): site for site in prebreakdown_df.file_name.unique()
    }
    site_no_name_dict_ = dict(sorted(site_no_name_dict_.items()))
    site_name_sorted_ = list(site_no_name_dict_.values())
    return site_name_sorted_


def plot_prebreakdown_box_plots(
    data, outfile, folder_path=path_box_fig, color_=None, hover_data_=[]
):
    xcat_order = get_correct_sort_order_site_name(data)
    fig = px.box(
        data, x="file_name", y="prebreakdown_vol", color=color_, hover_data=hover_data_,
    )
    fig.update_layout(
        font=dict(family="Arial", size=18,),
        xaxis={"categoryorder": "array", "categoryarray": xcat_order},
        yaxis=dict(range=[500, 2500]),
        hoverlabel=dict(bgcolor="white", font_size=18, font_family="Rockwell"),
    )
    fig.update_layout()
    fig.write_html(os.path.join(folder_path, f"{outfile}.html"))


def plot_pair_plots(
    data,
    outfile,
    folder_path,
    labels_=None,
    dimensions_=(
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_acceleration_lane",
        "mainline_aadt_2018",
        "breakdowns_by_tot",
        "ffs",
    ),
):
    fig2 = px.scatter_matrix(data, dimensions=dimensions_, labels=labels_,)
    fig2.show()
    fig2.write_html(os.path.join(folder_path, f"{outfile}.html"))


def save_cart(
    data,
    outfile,
    folder_path,
    x_vars,
    y_var="prebreakdown_vol",
    max_depth_=6,
):
    lb = preprocessing.LabelBinarizer()
    data.ramp_metering = lb.fit_transform(data.ramp_metering)
    y = data[y_var]
    X = data[[col for col in data.columns if col in x_vars]]
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
    graph.render(os.path.join(folder_path, f"{outfile}_{max_depth_}"))


if __name__ == "__main__":
    # Read data
    prebreakdown_df_all = pd.read_csv(path_prebreakdown_df_all)
    prebreakdown_df_merge_meta = pd.read_csv(path_prebreakdown_merge_and_meta)
    prebreakdown_df_ufl_meta = pd.read_csv(path_prebreakdown_df_ufl_meta)
    # Plot box plots
    site_name_sorted_all = get_correct_sort_order_site_name(
        prebreakdown_df=prebreakdown_df_all
    )
    plot_prebreakdown_box_plots(
        data=prebreakdown_df_all, color_="geometry_type", outfile="box_plots_all"
    )

    prebreakdown_df_all.geometry_type.unique()
    prebreakdown_df_simple_geom = prebreakdown_df_all.loc[
        lambda df: df.geometry_type.isin(
            ["Simple merge", "Simple diverge", "Simple ramp weave"]
        )
    ]
    plot_prebreakdown_box_plots(
        data=prebreakdown_df_simple_geom,
        color_="geometry_type",
        outfile="box_plots_bundle_1_2",
    )

    for geometry_type in prebreakdown_df_all.geometry_type.unique():
        prebreakdown_df_fil = prebreakdown_df_all.query(
            """geometry_type == @geometry_type"""
        )
        plot_prebreakdown_box_plots(
            data=prebreakdown_df_fil, outfile=f"box_plots_{geometry_type}"
        )

    plot_prebreakdown_box_plots(
        data=prebreakdown_df_merge_meta,
        outfile="box_plots_simple_merge",
        color_="ramp_metering",
        hover_data_=[
            "ffs",
            "fix_ffs",
            "number_of_mainline_lane_downstream",
            "length_of_acceleration_lane",
            "mainline_aadt_2018",
            "estimated_capacity_veh_hr_ln",
        ],
    )

    dimensions1 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_acceleration_lane",
        "mainline_aadt_2018",
        "breakdowns_by_tot",
        "ffs",
    )
    labels = {
        "prebreakdown_vol": "prebrkdn_vol",
        "prebreakdown_speed": "prebrkdn_spd",
        "length_of_acceleration_lane": "len_acc_ln",
        "mainline_aadt_2018": "aadt_2018",
        "breakdowns_by_tot": "brkdn/tot_cnt",
        "ffs": "ffs",
        "ramp_vol": "ramp_vol"
    }

    plot_pair_plots(
        data=prebreakdown_df_merge_meta,
        outfile="pair_plot_simple_merge",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions1,
    )
    save_cart(
        data=prebreakdown_df_merge_meta,
        outfile="cart_prebreakdown_simple_merge",
        folder_path=path_figures,
        y_var="prebreakdown_vol",
        x_vars=[
            "ffs",
            "breakdowns_by_tot",
            "number_of_mainline_lane_upstream",
            "number_of_on_ramp_lanes_at_ramp_terminal",
            "ramp_metering",
            "length_of_acceleration_lane",
            "prebreakdown_speed",
            "mainline_aadt_2018",
        ],
        max_depth_=4,
    )


    dimensions2 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_acceleration_lane",
        "mainline_aadt_2018",
        "breakdowns_by_tot",
        "ramp_vol",
        "ffs",
    )
    plot_pair_plots(
        data=prebreakdown_df_ufl_meta,
        outfile="pair_plot_simple_merge_ufl",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions2,
    )

    save_cart(
        data=prebreakdown_df_merge_meta,
        outfile="cart_prebreakdown_simple_merge_ufl",
        folder_path=path_figures,
        y_var="prebreakdown_vol",
        x_vars=[
            "ffs",
            "breakdowns_by_tot",
            "number_of_mainline_lane_upstream",
            "number_of_on_ramp_lanes_at_ramp_terminal",
            "ramp_metering",
            "length_of_acceleration_lane",
            "prebreakdown_speed",
            "mainline_aadt_2018",
            "ramp_vol"
        ],
        max_depth_=4,
    )
