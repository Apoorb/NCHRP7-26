import os
import pandas as pd

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
path_figures = os.path.join(path_to_project, "figures")
path_cap_plots = os.path.join(path_figures, "cap_plots")
if not os.path.exists(path_cap_plots):
    os.mkdir(path_cap_plots)
path_prebreakdown_simple_merge = os.path.join(
    path_interim, "prebreakdown_df_all_merge_meta.csv"
)
path_cap_df_merge_and_meta = os.path.join(path_interim, "all_merge_meta.csv")
# path_prebreakdown_df_ufl_meta = os.path.join(
#     path_interim, "prebreakdown_ufl_merge_and_meta.csv"
# )
path_cap_diverge_and_meta = os.path.join(path_interim, "all_diverge_meta.csv")
path_cap_weave_and_meta = os.path.join(path_interim, "all_weave_meta.csv")


def get_correct_sort_order_site_name(prebreakdown_df):
    site_no_name_dict_ = {
        int(site.split("_")[0]): site for site in prebreakdown_df.file_name.unique()
    }
    site_no_name_dict_ = dict(sorted(site_no_name_dict_.items()))
    site_name_sorted_ = list(site_no_name_dict_.values())
    return site_name_sorted_


def plot_cap_hist_plots(
    data, outfile, folder_path=path_cap_plots, color_=None, hover_data_=[]
):
    xcat_order = get_correct_sort_order_site_name(data)
    fig = px.histogram(
        data,
        x="file_name",
        y="estimated_capacity",
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
    fig2 = px.scatter_matrix(data, dimensions=dimensions_, labels=labels_, color=color_)
    fig2.write_html(os.path.join(folder_path, f"{outfile}.html"), auto_open=False)


if __name__ == "__main__":
    # Read data
    cap_merge_df = pd.read_csv(path_cap_df_merge_and_meta)
    cap_merge_df = cap_merge_df.assign(
        geometry_type=lambda df: df.file_name.str.split("_", expand=True)[
            1
        ].str.capitalize()
    )
    cap_merge_df_fil = cap_merge_df.loc[
        lambda df: (df.estimated_capacity <= 2600)
        & (df.geometry_type != "Ramp metered")
    ]

    # prebreakdown_df_ufl_meta = pd.read_csv(path_prebreakdown_df_ufl_meta)
    cap_diverge_df = pd.read_csv(path_cap_diverge_and_meta)
    cap_diverge_df = cap_diverge_df.assign(
        geometry_type=lambda df: df.file_name.str.split("_", expand=True)[
            1
        ].str.capitalize()
    )
    cap_diverge_df_fil = cap_diverge_df.loc[lambda df: (df.estimated_capacity <= 2600)]

    cap_weave_df = pd.read_csv(path_cap_weave_and_meta)
    cap_weave_df = cap_weave_df.assign(
        geometry_type=lambda df: df.file_name.str.split("_", expand=True)[
            1
        ].str.capitalize()
    )
    cap_weave_df_fil = cap_weave_df.loc[lambda df: (df.estimated_capacity <= 2600)]

    # Plot box plots
    site_name_sorted_all = get_correct_sort_order_site_name(
        prebreakdown_df=cap_merge_df
    )
    plot_cap_hist_plots(
        data=cap_merge_df_fil, color_="geometry_type", outfile="cap_hist_plot_merge"
    )
    plot_cap_hist_plots(
        data=cap_diverge_df, color_="geometry_type", outfile="cap_hist_plot_diverge"
    )
    plot_cap_hist_plots(
        data=cap_weave_df, color_="geometry_type", outfile="cap_hist_plot_weave"
    )

    dimensions1 = (
        "estimated_capacity",
        "number_of_mainline_lane_upstream",
        "length_of_acceleration_lane",
        "mainline_aadt",
        "mainline_speed_limit",
        "ffs",
        "ramp_metering",
    )
    labels = {
        "estimated_capacity": "estimated_capacity",
        "prebreakdown_speed": "prebrkdn_spd",
        "length_of_acceleration_lane": "len_acc_ln",
        "mainline_aadt": "aadt",
        "ffs_cap_df": "ffs",
        "number_of_on_ramp_lanes_at_ramp_terminal": "no_on_ramp_ln_terminal",
        "number_of_mainline_lane_upstream": "mainline_ln",
    }

    plot_pair_plots(
        data=cap_merge_df_fil,
        outfile="pair_plot_simple_merge_1",
        folder_path=path_cap_plots,
        labels_=labels,
        dimensions_=dimensions1,
        color_="file_name",
    )

    dimensions4 = (
        "estimated_capacity",
        "number_of_mainline_lane_upstream",
        "number_of_off_ramp_lane",
        "length_of_deceleration_lane",
        "mainline_aadt",
        "ffs_cap_df",
        "mainline_speed_limit",
    )
    cap_diverge_df_check = cap_diverge_df.loc[lambda df: df.ffs_cap_df <= 100]
    plot_pair_plots(
        data=cap_diverge_df_check,
        outfile="pair_plot_simple_diverge_1",
        folder_path=path_cap_plots,
        labels_=labels,
        dimensions_=dimensions4,
        color_="file_name",
    )

    dimensions6 = (
        "estimated_capacity",
        "short_length_ls_ft",
        "mainline_aadt",
        "ffs_cap_df",
        "mainline_speed_limit",
    )
    dimensions7 = (
        "estimated_capacity",
        "mainline_grade",
        "hv",
        "interchange_density",
        "ramp_metering",
    )
    cap_weave_df_check = cap_weave_df.loc[lambda df: df.ffs_cap_df <= 100]

    plot_pair_plots(
        data=cap_weave_df_check,
        outfile="pair_plot_simple_weave_1",
        folder_path=path_cap_plots,
        labels_=labels,
        dimensions_=dimensions6,
        color_="file_name",
    )
    plot_pair_plots(
        data=cap_weave_df_check,
        outfile="pair_plot_simple_weave_2",
        folder_path=path_cap_plots,
        labels_=labels,
        dimensions_=dimensions7,
        color_="file_name",
    )

    dimensions1 = (
        "estimated_capacity",
        "number_of_mainline_lane_upstream",
        "length_of_acceleration_lane",
        "mainline_aadt",
        "mainline_speed_limit",
        "ffs_cap_df",
        "ramp_metering",
    )

    x_train = cap_merge_df_fil.filter(
        items=[
            "ffs_cap_df",
            "length_of_acceleration_lane",
            "mainline_aadt",
            "ramp_metering",
        ]
    )
    y_train = cap_merge_df_fil.filter(items=["estimated_capacity"])
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_train)
    coeff_df = pd.DataFrame(regressor.coef_.T, x_train.columns, columns=["Coefficient"])

    import statsmodels.api as sm

    x_train_1 = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_train_1).fit()
    model.summary()

    ######################################################################################
    cap_merge_df_fil.columns
    import plotly.graph_objects as go

    for col in [
        "number_of_mainline_lane_downstream",
        "number_of_mainline_lane_upstream",
        "number_of_on_ramp_lanes_at_ramp_terminal",
        "ramp_metering",
    ]:
        fig1 = px.box(cap_merge_df_fil, x=col, y="estimated_capacity")
        data = fig1.data[0]
        data["boxmean"] = True
        fig2 = go.Figure(data)
        fig2.show()
        fig1.write_html(
            os.path.join(path_cap_plots, f"{col}_merge.html"), auto_open=False
        )

    for col in [
        "mainline_grade",
        "hv",
        "length_of_acceleration_lane",
        "ffs_cap_df",
        "mainline_aadt",
    ]:
        fig1 = px.scatter(cap_merge_df_fil, x=col, y="estimated_capacity")
        fig1.write_html(
            os.path.join(path_cap_plots, f"{col}_merge.html"), auto_open=False
        )

    path_useful_plots = os.path.join(path_cap_plots, "useful_plots")
    fig1 = px.scatter(
        cap_merge_df_fil.query("length_of_acceleration_lane <=1400"),
        x="length_of_acceleration_lane",
        y="estimated_capacity",
        trendline="ols",
        title="acc_len_1400_trend",
    )
    results = px.get_trendline_results(fig1)
    fig1.update_layout(height=800, width=800)
    fig1.write_html(
        os.path.join(path_useful_plots, "length_of_acceleration_lane_1400_merge.html"),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_merge_df_fil.query("length_of_acceleration_lane <=1500"),
        x="length_of_acceleration_lane",
        y="estimated_capacity",
        trendline="ols",
        title="acc_len_1500_trend",
    )
    results = px.get_trendline_results(fig1)
    fig1.update_layout(height=800, width=800)
    fig1.write_html(
        os.path.join(path_useful_plots, "length_of_acceleration_lane_1500_merge.html"),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_merge_df_fil.query("length_of_acceleration_lane <=1500"),
        x="mainline_speed_limit",
        y="estimated_capacity",
        facet_col="number_of_mainline_lane_downstream",
        color="length_of_acceleration_lane",
        trendline="ols",
        title="acc_len_1500_speed_impact",
    )
    fig1.show()
    fig1.write_html(
        os.path.join(path_useful_plots, "acc_len_1500_speed_impact.html"),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_merge_df_fil.query("length_of_acceleration_lane <=1500"),
        x="ffs_cap_df",
        y="estimated_capacity",
        facet_col="number_of_mainline_lane_downstream",
        color="length_of_acceleration_lane",
        trendline="ols",
        title="acc_len_1500_ffs_cap_df_impact",
    )
    fig1.show()
    fig1.write_html(
        os.path.join(path_useful_plots, "acc_len_1500_ffs_cap_df_impact.html"),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_merge_df_fil.query("length_of_acceleration_lane <=1500"),
        x="ffs_cap_df",
        y="estimated_capacity",
        facet_col="number_of_mainline_lane_downstream",
        color="length_of_acceleration_lane",
        trendline="ols",
        title="acc_len_1500_ffs_cap_df_impact",
    )
    fig1.show()
    fig1.write_html(
        os.path.join(path_useful_plots, "acc_len_1500_ffs_cap_df_impact.html"),
        auto_open=False,
    )

    #####################################################################################
    cap_diverge_df_fil.columns
    for col in [
        "number_of_mainline_lane_downstream",
        "number_of_mainline_lane_upstream",
        "number_of_off_ramp_lane",
    ]:
        fig1 = px.box(cap_diverge_df_fil, x=col, y="estimated_capacity", title=col)
        data = fig1.data[0]
        data["boxmean"] = True
        fig2 = go.Figure(data)
        fig1.write_html(
            os.path.join(path_cap_plots, f"{col}_diverge.html"), auto_open=False
        )

    for col in [
        "mainline_grade",
        "hv",
        "length_of_deceleration_lane",
        "ffs_cap_df",
        "mainline_aadt",
    ]:
        fig1 = px.scatter(cap_diverge_df_fil, x=col, y="estimated_capacity", title=col)
        fig1.write_html(
            os.path.join(path_cap_plots, f"{col}_diverge.html"), auto_open=False
        )

    path_useful_plots = os.path.join(path_cap_plots, "useful_plots")
    fig1 = px.scatter(
        cap_diverge_df_fil.query("length_of_deceleration_lane <=400"),
        x="length_of_deceleration_lane",
        y="estimated_capacity",
        trendline="ols",
        title="decc_len_400_trend",
    )
    results = px.get_trendline_results(fig1)
    fig1.update_layout(height=800, width=800)
    fig1.write_html(
        os.path.join(path_useful_plots, "length_of_deceleration_lane_400_diverge.html"),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_diverge_df_fil.query("length_of_deceleration_lane <=1500"),
        x="length_of_deceleration_lane",
        y="estimated_capacity",
        trendline="ols",
        title="decc_len_1500_trend",
    )
    results = px.get_trendline_results(fig1)
    fig1.update_layout(height=800, width=800)
    fig1.write_html(
        os.path.join(
            path_useful_plots, "length_of_deceleration_lane_1500_diverge.html"
        ),
        auto_open=False,
    )

    fig1 = px.scatter(
        cap_diverge_df_fil,
        x="mainline_speed_limit",
        y="estimated_capacity",
        facet_col="number_of_mainline_lane_downstream",
        color="length_of_deceleration_lane",
        trendline="ols",
        title="mainline_speed_limit_decc",
    )
    fig1.write_html(
        os.path.join(path_useful_plots, "decc_len_speed_impact.html"), auto_open=False
    )

    fig1 = px.scatter(
        cap_diverge_df_fil,
        x="ffs_cap_df",
        y="estimated_capacity",
        facet_col="number_of_mainline_lane_downstream",
        color="length_of_deceleration_lane",
        trendline="ols",
        title="ffs_cap_df_diverge",
    )
    fig1.show()
    fig1.write_html(
        os.path.join(path_useful_plots, "ffs_cap_df_diverge.html"), auto_open=False
    )

    #####################################################################################

    for col in [
        "mainline_grade",
        "hv",
        "short_length_ls_ft",
        "ffs_cap_df",
        "mainline_aadt",
    ]:
        fig1 = px.scatter(
            cap_weave_df_fil.query("short_length_ls_ft < 3000"),
            x=col,
            y="estimated_capacity",
            title=f"short_length_ls_ft_3000_{col}",
            trendline="ols",
            facet_col="geometry_type",
        )
        fig1.write_html(
            os.path.join(path_cap_plots, f"{col}_weave.html"), auto_open=False
        )
