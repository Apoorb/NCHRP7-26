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
    path_interim, "prebreakdown_df_all_merge_meta.csv"
)

if __name__ == "__main__":
    # Read data
    cap_merge_df = pd.read_csv(path_cap_df_merge_and_meta)
    cap_merge_df = cap_merge_df.assign(
        geometry_type=lambda df: df.file_name.str.split("_", expand=True)[
            1
        ].str.capitalize()
    )
    cap_merge_df_fil = cap_merge_df.loc[
        lambda df: (df.estimated_capacity <= 3000)
        & (df.geometry_type != "Ramp metered")
    ]
    cap_merge_df_fil_len_acc_1500 = cap_merge_df_fil.query(
        "length_of_acceleration_lane <=1500"
    )

    prebreakdown_merge = pd.read_csv(path_prebreakdown_merge)
    prebreakdown_merge_len_acc_1500 = prebreakdown_merge.loc[
        lambda df: (df.length_of_acceleration_lane <= 1500)
        & (df.estimated_capacity <= 3000)
        & (df.mainline_vol <= 3000)
        & (df.mainline_vol >= 1000)
        & (df.geometry_type.isin(["Close merge", "Simple merge", "Two lane on ramp"]))
    ]
    # prebreakdown_merge_len_acc_1500 = (
    #     prebreakdown_merge_len_acc_1500.assign(Time=lambda df: pd.to_datetime(df.Time))
    #     .set_index("Time")
    # )
    # prebreakdown_merge_len_acc_1500 = pd.concat(
    #     [
    #     prebreakdown_merge_len_acc_1500.between_time("6:00", "10:30"),
    #     prebreakdown_merge_len_acc_1500.between_time("3:00", "8:30")
    #     ]
    # ).reset_index()

    x_vars = [
        "length_of_acceleration_lane",
        "fwy_to_fwy_ramp",
        "number_of_on_ramp_lane_gore",
        "geometry_type",
        "number_of_mainline_lane_downstream",
        "ramp_metering",
        "ffs_cap_df",
    ]

    prebreakdown_merge_len_acc_1500 = prebreakdown_merge_len_acc_1500.assign(
        fwy_to_fwy_ramp=lambda df: np.select(
            [df.fwy_to_fwy_ramp == "no", df.fwy_to_fwy_ramp == "yes"], [0, 1]
        ),
    )
    y_var = ["mainline_vol", "failure"]
    prebreakdown_merge_len_acc_1500_model_df = prebreakdown_merge_len_acc_1500.filter(
        items=x_vars + y_var
    )

    temp = pd.get_dummies(
        prebreakdown_merge_len_acc_1500_model_df.geometry_type
    ).rename(columns={"Close merge": "close_merge", "Simple merge": "simple_merge"})
    prebreakdown_merge_len_acc_1500_model_df_one_hot = pd.concat(
        [prebreakdown_merge_len_acc_1500_model_df.drop(columns="geometry_type"), temp],
        axis=1,
    )

    fig = px.histogram(
        prebreakdown_merge_len_acc_1500_model_df_one_hot,
        x="mainline_vol",
        color="failure",
    )
    prebreakdown_merge_len_acc_1500_model_df_one_hot_no_censor = prebreakdown_merge_len_acc_1500_model_df_one_hot.query(
        "failure==1"
    )

    aft = WeibullAFTFitter()

    aft.fit(
        prebreakdown_merge_len_acc_1500_model_df_one_hot,
        duration_col="mainline_vol",
        event_col="failure",
        formula="ramp_metering+length_of_acceleration_lane+ffs_cap_df+number_of_mainline_lane_downstream+simple_merge",
    )

    aft.print_summary()
    aft.plot()
    aft.median_survival_time_
    aft.mean_survival_time_

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    aft.plot_partial_effects_on_outcome(
        "ramp_metering",
        [0, 1],
        cmap="coolwarm",
        ax=ax,
        plot_baseline=False,
        times=range(1000, 3200, 50),
    )

    fig2, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    aft.plot_partial_effects_on_outcome(
        "length_of_acceleration_lane",
        [500, 1000, 1500],
        plot_baseline=False,
        cmap="coolwarm",
        ax=ax1,
        times=range(1000, 3200, 50),
    )

    fig2, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    aft.plot_partial_effects_on_outcome(
        "ffs_cap_df",
        [50, 55, 60, 65, 70, 75, 80],
        plot_baseline=False,
        cmap="coolwarm",
        ax=ax1,
        times=range(1000, 3200, 50),
    )

    fig2, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    aft.plot_partial_effects_on_outcome(
        "number_of_mainline_lane_downstream",
        [2, 3, 4, 5],
        plot_baseline=False,
        cmap="coolwarm",
        ax=ax1,
        times=range(1000, 3200, 50),
    )
