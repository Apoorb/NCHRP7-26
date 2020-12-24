import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn import tree
from sklearn import preprocessing
import graphviz
import plotly.io as pio
from src.utils import get_project_root
from sklearn.linear_model import LinearRegression
import statsmodels

pio.renderers.default = "browser"

# Set paths:
path_to_project = str(get_project_root())
path_interim = os.path.join(path_to_project, "data", "interim")
path_processed = os.path.join(path_to_project, "data", "processed")
path_figures_v1 = os.path.join(path_to_project, "figures_v1")
if not os.path.exists(path_figures_v1):
    os.mkdir(path_figures_v1)
path_plot_dump = os.path.join(path_figures_v1, "plot_dump")
if not os.path.exists(path_plot_dump):
    os.mkdir(path_plot_dump)
path_cap_weave_and_meta = os.path.join(path_interim, "all_weave_meta.csv")


def plot_scatter(data, x_, y_, title_addon, facet_, color_, save_dir, height_, width_):
    fig1 = px.scatter(
        data,
        x=x_,
        y=y_,
        facet_col=facet_,
        color=color_,
        trendline="ols",
        title=f"{x_}_{y_}_{title_addon}_{facet_}",
    )
    fig1.update_layout(uniformtext_minsize=14, uniformtext_mode="hide")
    fig1.write_html(
        os.path.join(save_dir, f"{x_}_{y_}_{title_addon}_{facet_}.html"),
        auto_open=False,
    )


def plot_box(data, x_, y_, title_addon, facet_, color_, save_dir, height_, width_):
    fig1 = px.box(
        data,
        x=x_,
        y=y_,
        facet_col=facet_,
        color=color_,
        title=f"{x_}_{y_}_{title_addon}_{facet_}",
    )
    fig1.update_traces(boxmean=True)
    fig1.update_layout(font_size=16, height=height_, width=width_)
    fig1.write_html(
        os.path.join(save_dir, f"{x_}_{y_}_{title_addon}_{facet_}.html"),
        auto_open=False,
    )


if __name__ == "__main__":
    # Read data
    cap_weave_df = pd.read_csv(path_cap_weave_and_meta)
    cap_weave_df = cap_weave_df.assign(
        geometry_type=lambda df: df.file_name.str.split("_", expand=True)[
            1
        ].str.capitalize()
    )
    cap_weave_df_fil = cap_weave_df.loc[lambda df: (df.estimated_capacity <= 3000)]

    cont_vars = [
        "short_length_ls_ft",
        "mainline_grade",
        "mainline_aadt",
        "interchange_density",
        "dist_to_upstream_ramp_ft",
        "dist_to_downstream_ramp_ft",
        "ffs_cap_df",
    ]
    ordinal_vars = [
        "number_of_mainline_lane_downstream",
        "number_of_mainline_lane_upstream",
        "num_of_mainline_lane_weaving_section",
        "number_of_off_ramp_lane",
        "presence_of_adjacent_ramps",
        "mainline_speed_limit",
        "n_wl",
        "lcrf",
        "lcfr",
        "lcrr",
    ]
    nominal_vars = [
        "area_type",
        "upstream_ramp_type_on_off",
        "downstream_ramp_type_on_off",
        "geometry_type",
    ]

    process_var = [
        "fwy_to_fwy_ramp_on",
        "signal_ramp_terminal_on",
        "free_flow_ramp_terminal_on",
        "roundabout_terminal_on",
        "fwy_to_fwy_ramp_off",
        "signal_ramp_terminal_off",
        "free_flow_ramp_terminal_off",
        "roundabout_terminal_off",
    ]

    path_weave_plot_dump = os.path.join(path_plot_dump, "weave_plot_dump")
    if not os.path.exists(path_weave_plot_dump):
        os.mkdir(path_weave_plot_dump)

    plot_scatter(
        data=cap_weave_df_fil,
        x_="ffs_cap_df",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=800,
        width_=1200,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="fwy_to_fwy_ramp_on",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="signal_ramp_terminal_on",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="free_flow_ramp_terminal_on",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="fwy_to_fwy_ramp_off",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="signal_ramp_terminal_off",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="free_flow_ramp_terminal_off",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="n_wl",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="geometry_type",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="num_of_mainline_lane_weaving_section",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="number_of_mainline_lane_downstream",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_box(
        data=cap_weave_df_fil,
        x_="number_of_mainline_lane_upstream",
        y_="estimated_capacity",
        title_addon="",
        facet_=None,
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=600,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="ffs_cap_df",
        y_="estimated_capacity",
        title_addon="",
        facet_="num_of_mainline_lane_weaving_section",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=800,
        width_=1200,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="mainline_speed_limit",
        y_="estimated_capacity",
        title_addon="",
        facet_="num_of_mainline_lane_weaving_section",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=800,
        width_=1200,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="mainline_speed_limit",
        y_="ffs_cap_df",
        title_addon="",
        facet_="num_of_mainline_lane_weaving_section",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=1000,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="mainline_speed_limit",
        y_="estimated_capacity",
        title_addon="",
        facet_="num_of_mainline_lane_weaving_section",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=600,
        width_=1000,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="short_length_ls_ft",
        y_="estimated_capacity",
        title_addon="",
        facet_="geometry_type",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=800,
        width_=1200,
    )

    plot_scatter(
        data=cap_weave_df_fil,
        x_="interchange_density",
        y_="estimated_capacity",
        title_addon="",
        facet_="geometry_type",
        color_=None,
        save_dir=path_weave_plot_dump,
        height_=800,
        width_=1200,
    )
