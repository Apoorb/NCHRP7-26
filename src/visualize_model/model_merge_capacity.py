"""
Fit accelerated failure time model on the merge uncongested and pre-breakdown data.
"""
import os
import pandas as pd
from lifelines import WeibullAFTFitter
from src.utils import get_project_root
import plotly.express as px
import plotly.io as pio
import numpy as np
from matplotlib import pyplot as plt

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
path_cap_df_merge_and_meta = os.path.join(path_interim, "all_merge_meta.csv")
path_prebreakdown_merge = os.path.join(
    path_interim, "prebkdn_uncongested_merge_meta.csv"
)


def one_hot_coding_cat(data):
    hot_code_df = pd.get_dummies(data.geometry_type)
    hot_code_df = hot_code_df.rename(
        columns={col: str.lower(col.replace(" ", "_")) for col in hot_code_df.columns}
    )
    data_one_hot_code_df = pd.concat([data, hot_code_df], axis=1,)
    return data_one_hot_code_df


def fit_aft_model(data, formula_, yvar_="mainline_vol", event_var="failure"):
    aft = WeibullAFTFitter()
    aft.fit(
        data, duration_col=yvar_, event_col=event_var, formula=formula_,
    )
    return aft


def aft_plots_on_survival(model, var, var_levels, ax):
    model.plot_partial_effects_on_outcome(
        var,
        var_levels,
        cmap="coolwarm",
        ax=ax,
        plot_baseline=False,
        times=range(1000, 3200, 50),
    )


if __name__ == "__main__":

    prebreakdown_merge = pd.read_csv(path_prebreakdown_merge)
    prebreakdown_merge_vol_fil = prebreakdown_merge.loc[
        lambda df: (df.mainline_vol >= 1000)
        & (df.mainline_vol <= 3000)
        & (df.geometry_type.isin(["Close merge", "Simple merge", "Two lane on ramp"]))
    ]
    prebreakdown_merge_len_acc_1500 = prebreakdown_merge_vol_fil.loc[
        lambda df: (df.length_of_acceleration_lane <= 1500)
    ]

    # Columns in prebreakdown_merge_vol_fil that can be used for AFT
    # x_vars = [
    #     "length_of_acceleration_lane",
    #     "fwy_to_fwy_ramp",
    #     "number_of_on_ramp_lane_gore",
    #     "geometry_type",
    #     "number_of_mainline_lane_downstream",
    #     "ramp_metering",
    #     "ffs_cap_df",
    # ]

    # Use one hot coding on categorical variables.
    prebreakdown_merge_vol_fil = one_hot_coding_cat(prebreakdown_merge_vol_fil)
    prebreakdown_merge_len_acc_1500 = one_hot_coding_cat(
        prebreakdown_merge_len_acc_1500
    )

    formula_ = (
        "ramp_metering+length_of_acceleration_lane+ffs_cap_df"
        "+number_of_mainline_lane_downstream+simple_merge"
    )
    # Fit accelerated failure time model.
    aft_all_data = fit_aft_model(prebreakdown_merge_vol_fil, formula_=formula_)
    aft_all_data.print_summary(decimals=5)
    aft_all_data.median_survival_time_
    # Fit accelerated failure time model with just the intercept.
    formula_ = "1"
    aft_intercept = fit_aft_model(prebreakdown_merge_vol_fil, formula_=formula_)
    aft_intercept.print_summary()
    aft_all_data.log_likelihood_ratio_test()

    var_levels_dict = {
        "ramp_metering": [0, 1],
        "length_of_acceleration_lane": [100, 500, 1000, 1500],
        "ffs_cap_df": [55, 60, 65, 70, 75, 80],
        "number_of_mainline_lane_downstream": [2, 3, 4, 5],
        "simple_merge": [0, 1],
    }
    var_iter = var_levels_dict.__iter__()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i, ax in enumerate(fig.axes):
        var = var_iter.__next__()
        var_levels = var_levels_dict[var]
        aft_plots_on_survival(model=aft_all_data, var=var, var_levels=var_levels, ax=ax)
        if var == "simple_merge":
            break

    ax.set_xlabel("Volume (veh/hr/ln)")
    fig.text(0.5, 0.04, "Volume (veh/hr/ln)", ha="center")
    fig.text(0.04, 0.5, "1 - Breakdown Probability", va="center", rotation="vertical")
    fig.savefig(
        os.path.join(path_figures_v1, "survival_plot_merge_all_usable_site.jpg")
    )
    sum_df = prebreakdown_merge_vol_fil.drop_duplicates(
        "file_name"
    ).geometry_type.value_counts()
    sum_df.to_csv(
        os.path.join(path_figures_v1, "survival_plot_merge_all_usable_site.csv")
    )

    formula_ = (
        "ramp_metering+length_of_acceleration_lane+ffs_cap_df+fwy_to_fwy_ramp"
        "+number_of_mainline_lane_downstream+simple_merge"
    )
    aft_all_1500_acc = fit_aft_model(prebreakdown_merge_len_acc_1500, formula_=formula_)
    aft_all_1500_acc.print_summary(decimals=5)

    formula_ = "1"
    aft_intercept = fit_aft_model(prebreakdown_merge_vol_fil, formula_=formula_)
    aft_intercept.print_summary()
