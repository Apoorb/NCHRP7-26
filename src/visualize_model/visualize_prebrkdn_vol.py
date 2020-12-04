import os
import pandas as pd

import plotly.express as px
from sklearn import tree
from sklearn import preprocessing
import graphviz
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
path_prebreakdown_simple_merge = os.path.join(
    path_interim, "prebreakdown_df_all_merge_meta.csv"
)
path_prebreakdown_df_merge_and_meta = os.path.join(
    path_interim, "prebreakdown_df_all_merge_meta.csv"
)
# path_prebreakdown_df_ufl_meta = os.path.join(
#     path_interim, "prebreakdown_ufl_merge_and_meta.csv"
# )
path_prebreakdown_diverge_and_meta = os.path.join(path_interim, "cap_weave_df.csv")
path_prebreakdown_weave_and_meta = os.path.join(path_interim, "cap_weave_df.csv")


def filter_to_prebreakdown_data(data):
    data_fil = data.loc[lambda df: df.failure == 1]
    data_fil = data_fil.rename(
        columns={
            "mainline_vol": "prebreakdown_vol",
            "mainline_speed": "prebreakdown_speed",
        }
    )
    return data_fil


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
    data_fil = filter_to_prebreakdown_data(data)
    xcat_order = get_correct_sort_order_site_name(data)
    fig = px.box(
        data_fil,
        x="file_name",
        y="prebreakdown_vol",
        color=color_,
        hover_data=hover_data_,
    )
    fig.update_layout(
        font=dict(family="Arial", size=18,),
        xaxis={"categoryorder": "array", "categoryarray": xcat_order},
        yaxis=dict(range=[500, 2500]),
        hoverlabel=dict(bgcolor="white", font_size=18, font_family="Rockwell"),
    )
    fig.update_layout()
    fig.write_html(os.path.join(folder_path, f"{outfile}.html"), auto_open=False)


def plot_pair_plots(data, outfile, folder_path, dimensions_, labels_=None, color_=None):
    data_fil = filter_to_prebreakdown_data(data)
    fig2 = px.scatter_matrix(
        data_fil, dimensions=dimensions_, labels=labels_, color=color_
    )
    fig2.write_html(os.path.join(folder_path, f"{outfile}.html"), auto_open=False)


def save_cart(
    data, outfile, folder_path, x_vars, y_var="prebreakdown_vol", max_depth_=6,
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
    prebreakdown_df_merge_and_meta = pd.read_csv(path_prebreakdown_df_merge_and_meta)
    # prebreakdown_df_ufl_meta = pd.read_csv(path_prebreakdown_df_ufl_meta)
    prebreakdown_diverge_and_meta = pd.read_csv(path_prebreakdown_diverge_and_meta)
    prebreakdown_weave_and_meta = pd.read_csv(path_prebreakdown_weave_and_meta)

    # Plot box plots
    site_name_sorted_all = get_correct_sort_order_site_name(
        prebreakdown_df=prebreakdown_df_merge_and_meta
    )
    plot_prebreakdown_box_plots(
        data=prebreakdown_df_merge_and_meta,
        color_="geometry_type",
        outfile="box_plot_merge",
    )
    plot_prebreakdown_box_plots(
        data=prebreakdown_diverge_and_meta,
        color_="geometry_type",
        outfile="box_plot_diverge",
    )
    plot_prebreakdown_box_plots(
        data=prebreakdown_weave_and_meta,
        color_="geometry_type",
        outfile="box_plot_weave",
    )

    dimensions1 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_acceleration_lane",
        "mainline_aadt",
        "breakdown_events",
        "ffs",
    )
    dimensions2 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "mainline_grade",
        "hv",
        "number_of_mainline_lane_upstream",
        "number_of_on_ramp_lanes_at_ramp_terminal",
        "ramp_metering",
    )
    labels = {
        "prebreakdown_vol": "prebrkdn_vol",
        "prebreakdown_speed": "prebrkdn_spd",
        "length_of_acceleration_lane": "len_acc_ln",
        "mainline_aadt": "aadt",
        "ffs": "ffs",
        "number_of_on_ramp_lanes_at_ramp_terminal": "no_on_ramp_ln_terminal",
        "number_of_mainline_lane_upstream": "mainline_ln",
    }

    plot_pair_plots(
        data=prebreakdown_df_merge_and_meta,
        outfile="pair_plot_simple_merge_1",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions1,
        color_="file_name",
    )
    plot_pair_plots(
        data=prebreakdown_df_merge_and_meta,
        outfile="pair_plot_simple_merge_2",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions2,
        color_="file_name",
    )
    dimensions4 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "length_of_deceleration_lane",
        "mainline_aadt",
        "ffs",
    )
    dimensions5 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "mainline_grade",
        "hv",
        "number_of_mainline_lane_upstream",
        "number_of_off_ramp_lane",
    )
    plot_pair_plots(
        data=prebreakdown_diverge_and_meta,
        outfile="pair_plot_simple_diverge_1",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions4,
        color_="file_name",
    )
    plot_pair_plots(
        data=prebreakdown_diverge_and_meta,
        outfile="pair_plot_simple_diverge_2",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions5,
        color_="file_name",
    )

    dimensions6 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "short_length_ls_ft",
        "mainline_aadt",
        "ffs",
    )
    dimensions7 = (
        "prebreakdown_vol",
        "prebreakdown_speed",
        "mainline_grade",
        "hv",
        "interchange_density",
        "ramp_metering",
    )
    plot_pair_plots(
        data=prebreakdown_weave_and_meta,
        outfile="pair_plot_simple_weave_1",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions6,
        color_="file_name",
    )
    plot_pair_plots(
        data=prebreakdown_weave_and_meta,
        outfile="pair_plot_simple_weave_2",
        folder_path=path_figures,
        labels_=labels,
        dimensions_=dimensions7,
        color_="file_name",
    )
    #
    # # fig=px.box(cap_weave_df,
    # #        x="short_length_ls_ft",
    # #        y="prebreakdown_vol")
    # # fig.show()
    #
    # save_cart(
    #     data=prebreakdown_df_ufl_meta,
    #     outfile="cart_prebreakdown_simple_merge_ufl",
    #     folder_path=path_figures,
    #     y_var="prebreakdown_vol",
    #     x_vars=[
    #         "ffs",
    #         "breakdowns_by_tot",
    #         "number_of_mainline_lane_upstream",
    #         "number_of_on_ramp_lanes_at_ramp_terminal",
    #         "ramp_metering",
    #         "length_of_acceleration_lane",
    #         "prebreakdown_speed",
    #         "mainline_aadt_2018",
    #         "ramp_vol",
    #     ],
    #     max_depth_=3,
    # )
    # save_cart(
    #     data=prebreakdown_df_simple_merge,
    #     outfile="cart_prebreakdown_simple_merge",
    #     folder_path=path_figures,
    #     y_var="prebreakdown_vol",
    #     x_vars=[
    #         "ffs",
    #         "breakdowns_by_tot",
    #         "number_of_mainline_lane_upstream",
    #         "number_of_on_ramp_lanes_at_ramp_terminal",
    #         "ramp_metering",
    #         "length_of_acceleration_lane",
    #         "prebreakdown_speed",
    #         "mainline_aadt_2018",
    #     ],
    #     max_depth_=3,
    # )
