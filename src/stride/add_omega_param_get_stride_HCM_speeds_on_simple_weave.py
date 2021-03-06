"""
Script for:
    reading the uncongested volume and speed data along with metadata.
    Estimating HCM speed
    Estimating STRIDE speed with Nagui's parameter
    Estimating STRIDE parmaeters and then estimating speed---add omega to Vfr.
    Plotting HCM, STRIDE, and calibrated STRIDE speeds.
Created by: Apoorba Bibeka
Updated on: 12/24/2020
"""
import os
import pandas as pd
from src.utils import get_project_root
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import math

pio.renderers.default = "browser"

# Resources/ general overview

# 1 os: I use this to stitch togaher paths.

# 2 pandas: my bread and butter. The best way to learn it would be to read the pandas
# documentation. Following are some resources that I have read:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
#  https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
# Pandas assign, rename, filter, drop, loc, query, datetime, and agg functions.

# 3: import plotly.io as pio
# import plotly.express as px
# from plotly.subplots import make_subplots
# from plotly.offline import plot
# import plotly.graph_objects as go
# Start here for learning plotly: https://plotly.com/python/plotly-express/
# I perfer it over seaborn and matplotlib for plotting as I generally able to do more
# here with less effort.

# 4: from scipy.optimize import curve_fit: scientific python for curve fitting.
# 5:  from sklearn.metrics import mean_squared_error:  Get mean squared error. One stop
# shop for most of the machine learning algorithm.

# Set paths:
path_to_project = str(get_project_root())
path_raw = os.path.join(path_to_project, "data", "raw")
path_interim = os.path.join(path_to_project, "data", "interim")
path_processed = os.path.join(path_to_project, "data", "processed")
path_figures_v1 = os.path.join(path_to_project, "figures_v1")
if not os.path.exists(path_figures_v1):
    os.mkdir(path_figures_v1)
path_plot_dump = os.path.join(path_figures_v1, "plot_dump")
if not os.path.exists(path_plot_dump):
    os.mkdir(path_plot_dump)
# I used prebreakdown data, uncongested data, metadata for attributes, capacity data to
# create prebkdn_uncongested_weave_and_meta.csv in the clean_prebreakdown_data.py.
# clean_prebreakdown_data.py also handles duplicated pre-breakdown and uncongested volumes
path_volume_weave = os.path.join(path_interim, "prebkdn_uncongested_weave_and_meta.csv")
# Use the STRIDE data that Lilian prepared for Vfr, Vrf, and FFS for now; we haven't
# decided what FFS to use yet (12/24/2020).
path_stride_streetlight_df = os.path.join(
    path_raw, "STRIDE Weaving Capacity_20201223.xlsx"
)
# Read the Vfr and Vrf data from Lilian's STRIDE method spreadsheet.
vfr_rf_df = (
    pd.read_excel(path_stride_streetlight_df, skiprows=2, sheet_name="4_Calibration")
    .rename(
        columns={
            "Site ID": "file_name",
            "Time Period": "time_period",
            "FFS": "FFS_lilian",
            "Vrf%": "Vrf_percent",
            "Vfr%": "Vfr_percent",
            "Vrr%": "Vrr_percent",
        }
    )
    .loc[lambda df: ~df.Vrf_percent.isna()]
    .assign(
        file_name=lambda df: df.file_name.str.strip().str.title(),
        start_time=lambda df: (
            df.time_period.str.split(r"[(]|[)]", expand=True)[1].str.split(
                "-", expand=True
            )[0]
        ),
        end_time=lambda df: (
            df.time_period.str.split(r"[(]|[)]", expand=True)[1].str.split(
                "-", expand=True
            )[1]
        ),
    )
    .assign(
        start_hour=lambda df: pd.to_datetime(df.start_time, format="%I%p").dt.hour,
        end_hour=lambda df: pd.to_datetime(df.end_time, format="%I%p").dt.hour,
    )
    .filter(
        items=[
            "file_name",
            "time_period",
            "Vrf_percent",
            "Vfr_percent",
            "Vrr_percent",
            "start_hour",
            "end_hour",
            "FFS_lilian",
        ]
    )
)
# Read the combined data for all weave sites and the metadata (output from the
# clean_prebreakdown_data.py).
vol_weave_df = pd.read_csv(path_volume_weave)
vol_weave_df_simple = vol_weave_df.query("geometry_type == 'Simple ramp weave'")
vol_weave_df_simple.file_name.unique()
# Keep the sites that Lilian has in the STRIDE spreadsheet
keep_files = [
    "50_Simple Ramp Weave_1",
    "51_Simple Ramp Weave_2",
    "52_Simple Ramp Weave_3",
    "53_Simple Ramp Weave_4",
    "55_Simple Ramp Weave_6",
    "56_Simple Ramp Weave_7",
    "57_Simple Ramp Weave_8",
    "58_Simple Ramp Weave_9",
    "59_Simple Ramp Weave_10",
    "61_Simple Ramp Weave_12",
    "62_Simple Ramp Weave_13",
    "63_Simple Ramp Weave_14",
    "64_Simple Ramp Weave_15",
]
vol_weave_df_simple_fil = vol_weave_df_simple.loc[
    lambda df: df.file_name.isin(keep_files)
]
# Check if there is extra sites or fewer sites in the STRIDE spreadsheet.
# Both should result in empty sets.
set(vfr_rf_df.file_name) - set(keep_files)
set(keep_files) - set(vfr_rf_df.file_name)
# Add an hour column to the weaving data to filter the time periods based on the
# STRIDE spreadsheet.
vol_weave_df_simple_fil.loc[:, "hour"] = pd.to_datetime(
    vol_weave_df_simple_fil.Time
).dt.hour
vol_weave_df_simple_fil_stride_hcm = (
    vol_weave_df_simple_fil.merge(
        vfr_rf_df,
        left_on=["file_name", "hour"],
        right_on=["file_name", "start_hour"],
        how="right",
    )
    .sort_values(["file_name", "Time"])
    .filter(
        items=[
            "file_name",
            "Time",
            "start_hour",
            "mainline_vol",
            "mainline_speed",
            "mainline_speed_limit",
            "FFS_lilian",
            "ffs_cap_df",
            "short_length_ls_ft",
            "Vrf_percent",
            "Vfr_percent",
            "num_of_mainline_lane_weaving_section",
            "mainline_grade",
            "hv",
            "lcfr",
            "lcrf",
            "n_wl",
            "interchange_density",
        ]
    )
    .drop(
        columns="ffs_cap_df"
    )  # dropping the FFS here and using the value from STRIDE spreadsheet
    # ffs_cap_df in drop function comes from the data output generated by Azy. We have
    # not finalized it yet, so I am using FFS values from the STRIDE spreadsheet.
    # Fixme: Figure out later what we are doing with FFS.
    .assign(
        ET=lambda df: np.select(
            [df.mainline_grade <= 2, df.mainline_grade > 2], [2, 3]
        ),
        f_hv=lambda df: 1 / (1 + ((df.hv / 100) * (df.ET - 1))),
        mainline_vol_pcu=lambda df: df.mainline_vol / df.f_hv,
    )
)
# QAQC dataframe
vol_weave_df_simple_fil_stride_unique = vol_weave_df_simple_fil_stride_hcm.drop_duplicates(
    "file_name"
)
# Get the required fields for applying STRIDE method
d_c_hcm_basic_seg = 45
a_hcm_basic_seg = 2
vol_weave_df_simple_fil_stride_hcm_extra_cols = vol_weave_df_simple_fil_stride_hcm.assign(
    Vrf=lambda df: df.Vrf_percent
    * df.mainline_vol_pcu
    * df.num_of_mainline_lane_weaving_section,
    Vfr=lambda df: df.Vfr_percent
    * df.mainline_vol_pcu
    * df.num_of_mainline_lane_weaving_section,
    c_hcm_basic_seg=lambda df: np.select(
        [
            (2200 + 10 * (df.FFS_lilian - 50)) <= 2400,
            (2200 + 10 * (df.FFS_lilian - 50)) > 2400,
        ],
        [2200 + 10 * (df.FFS_lilian - 50), 2400],
    ),
    bp_hcm_basic=lambda df: 1000 + 40 * (75 - df.FFS_lilian),
    s_hcm_basic=lambda df: np.select(
        [
            df.mainline_vol_pcu <= df.bp_hcm_basic,
            (df.mainline_vol_pcu > df.bp_hcm_basic)
            & (df.mainline_vol_pcu <= df.c_hcm_basic_seg),
            df.mainline_vol_pcu > df.c_hcm_basic_seg,
        ],
        [
            df.FFS_lilian,
            df.FFS_lilian
            - (
                (df.FFS_lilian - (df.c_hcm_basic_seg / d_c_hcm_basic_seg))
                * ((df.mainline_vol_pcu - df.bp_hcm_basic) ** a_hcm_basic_seg)
                / ((df.c_hcm_basic_seg - df.bp_hcm_basic) ** a_hcm_basic_seg)
            ),
            np.nan,
        ],
    ),
).loc[
    lambda df: (df.mainline_speed >= 0.7 * df.FFS_lilian)
    & (df.mainline_vol_pcu > 500)
    # & (df.FFS_lilian <= 75)
    & (~df.s_hcm_basic.isna())
]
# Compute S_knot from STRIDE method with Nagui's calibrated parameters
# STRIDE calibrated parameters from Nagui's research
alpha = 0.025
beta = 17.302
gamma = 0.344
delta = 0.369
epsilon = 3


def get_S_not_STRIDE(row, alpha, beta, gamma, epsilon, delta, omega=1):
    Vrf = row.Vrf
    Vfr = row.Vfr
    nl = row.num_of_mainline_lane_weaving_section
    mainline_vol_pcu = row.mainline_vol_pcu
    short_length_ls_ft = row.short_length_ls_ft
    s_hcm_basic = row.s_hcm_basic
    return (
        s_hcm_basic
        - alpha
        * (((beta * Vrf + omega * Vfr) / (nl) ** epsilon) ** gamma)
        * (mainline_vol_pcu - 500)
        * (1 / short_length_ls_ft) ** delta
    )


vol_weave_df_simple_fil_stride_hcm_extra_cols.loc[
    :, "S_not_stride"
] = vol_weave_df_simple_fil_stride_hcm_extra_cols.apply(
    get_S_not_STRIDE,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    epsilon=epsilon,
    delta=delta,
    axis=1,
)
# QAQC dataframe
test_stride_speed_calc = vol_weave_df_simple_fil_stride_hcm_extra_cols.drop_duplicates(
    "file_name"
)

# Get HCM Predicted Speed for Weaving Sections
vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps = vol_weave_df_simple_fil_stride_hcm_extra_cols.assign(
    lc_min=lambda df: (df.lcfr * df.Vfr) + (df.lcrf * df.Vrf),
    vr=lambda df: (df.Vfr + df.Vrf)
    / (df.mainline_vol_pcu * df.num_of_mainline_lane_weaving_section),
    l_max=lambda df: (5728 * (1 + df.vr) ** 1.6 - (1566 * df.n_wl)),
    does_hcm_says_use_merge_diverge=lambda df: df.short_length_ls_ft > df.l_max,
    c_iwl=lambda df: df.c_hcm_basic_seg
    - (438.2 * (1 + df.vr) ** 1.6)
    + (0.0765 * df.short_length_ls_ft)
    + (119.8 * df.n_wl),
    c_iw=lambda df: np.select(
        [df.n_wl == 2, df.n_wl == 3],
        [2400 / df.vr, 3500 / df.vr],
        np.nan,  # n_wl should be 2 or 3 for this equation to work.
    ),
    c_iw_per_ln=lambda df: df.c_iw / df.num_of_mainline_lane_weaving_section,
    c_hcm_weave=lambda df: df[["c_iwl", "c_iw_per_ln"]].min(axis=1),
    v_by_c=lambda df: df.mainline_vol_pcu / df.c_hcm_weave,
    is_v_by_c_over_1=lambda df: df.v_by_c > 1,
    v_nw=lambda df: (df.mainline_vol_pcu * df.num_of_mainline_lane_weaving_section)
    - (df.Vfr + df.Vrf),
    lc_w=lambda df: df.lc_min
    + 0.39
    * (df.short_length_ls_ft - 300) ** 0.5
    * df.num_of_mainline_lane_weaving_section ** 2
    * (1 + df.interchange_density) ** 0.8,
    i_nw=lambda df: df.short_length_ls_ft * df.interchange_density * df.v_nw / 10000,
    lc_nw1=lambda df: (0.206 * df.v_nw)
    + (0.542 * df.short_length_ls_ft)
    - (192.6 * df.num_of_mainline_lane_weaving_section),
    lc_nw2=lambda df: 2135 + 0.223 * (df.v_nw - 2000),
    lc_nw3=lambda df: df.lc_nw1 + (df.lc_nw2 - df.lc_nw1) * (df.i_nw - 1300) / 650,
    lc_nw=lambda df: np.select(
        [
            df.lc_nw1 >= df.lc_nw2,
            df.i_nw <= 1300,
            df.i_nw >= 1950,
            (df.i_nw > 1300) & (df.i_nw < 1950),
        ],
        [df.lc_nw2, df.lc_nw1, df.lc_nw2, df.lc_nw3],
        np.nan,
    ),
    lc_all=lambda df: df.lc_w + df.lc_nw,
    s_min=15,
    s_max=lambda df: df.FFS_lilian,
    w=lambda df: 0.226 * (df.lc_all / df.short_length_ls_ft) ** 0.789,
    s_w=lambda df: df.s_min + (df.s_max - df.s_min) / (1 + df.w),
    s_nw=lambda df: df.FFS_lilian
    - (0.0072 * df.lc_min)
    - (0.0048 * df.mainline_vol_pcu),
    v_w=lambda df: (df.Vfr + df.Vrf),
    s_weave_hcm=lambda df: (df.v_w + df.v_nw)
    / ((df.v_w / df.s_w) + (df.v_nw / df.s_nw)),
    d_weave_hcm=lambda df: df.mainline_vol_pcu / df.s_weave_hcm,
)
test_hcm_speed_calc = vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.drop_duplicates(
    "file_name"
)
# QAQCed with Lilian's excel hcm capacity and speed calculations.

# Get RMSE
rmse_stride_speed = math.sqrt(
    mean_squared_error(
        vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.mainline_speed,
        vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.S_not_stride,
    )
)
rmse_hcm_speed = math.sqrt(
    mean_squared_error(
        vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.mainline_speed,
        vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.s_weave_hcm,
    )
)
rmse_stride_speed = np.round(rmse_stride_speed, 2)
rmse_hcm_speed = np.round(rmse_hcm_speed, 2)
print(f"Root mean squared error for STRIDE method = {rmse_stride_speed}")
print(f"Root mean squared error for HCM method = {rmse_hcm_speed}")

# Calibrating the STRIDE Parameters.
##########################################################################################
# Filter to columns that are needed for the STRIDE equation.
stride_model_fit_df = (
    vol_weave_df_simple_fil_stride_hcm_extra_cols.assign(
        s_knot_minus_s_b=lambda df: df.mainline_speed - df.s_hcm_basic
    )
    .rename(columns={"num_of_mainline_lane_weaving_section": "nl"})
    .filter(  # The order on the variables below matter. They are being read by
        # curve_fit_stride in this order. Do not change the variable order below!
        items=[
            "s_knot_minus_s_b",
            "Vrf",
            "Vfr",
            "nl",
            "mainline_vol_pcu",
            "short_length_ls_ft",
        ]
    )
)

# Estimate the STRIDE parameters---unconstrained optimization---use omega for Vfr
# Check with Nagui if STIDE parameters need to be constrained
# for optimization


def curve_fit_stride(X, alpha, beta, gamma, epsilon, delta, omega):
    s_knot_minus_s_b, Vrf, Vfr, nl, mainline_vol_pcu, short_length_ls_ft = X
    return (
        s_knot_minus_s_b
        + alpha
        * (((beta * Vrf + omega * Vfr) / (nl) ** epsilon) ** gamma)
        * (mainline_vol_pcu - 500)
        * (1 / short_length_ls_ft) ** delta
    )


Y = np.transpose(np.zeros(len(stride_model_fit_df)))  # Target is zero error
X = np.array(stride_model_fit_df.values.T)
np.set_printoptions(suppress=True)  # Don't show scientific numbers.
popt, pcov = curve_fit(curve_fit_stride, X, Y, maxfev=5000)
alpha_optimal, beta_optimal, gamma_optimal, epsilon_optimal, delta_optimal, omega_optimal = popt
Error = curve_fit_stride(
    X, alpha_optimal, beta_optimal, gamma_optimal, epsilon_optimal, delta_optimal, omega_optimal
)
# RMSE
rmse_calibrated_stride = np.round(sum((Error ** 2) / len(Error)) ** 0.5, 2)
print(
    f"Root mean squared error for Calibrated STRIDE method = {rmse_calibrated_stride}"
)

vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.loc[
    :, "S_not_stride_with_unconstrained_calibrated_parameters"
] = vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.apply(
    get_S_not_STRIDE,
    alpha=alpha_optimal,
    beta=beta_optimal,
    gamma=gamma_optimal,
    epsilon=epsilon_optimal,
    delta=delta_optimal,
    omega=omega_optimal,
    axis=1,
)
#########################################################################################

# Plots the Stride data
#########################################################################################
# Plot observed speed vs. HCM estimated speed and observed speed vs. stride estimated
# speed.
# Create two subplots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        f"HCM Estimated Speed (RMSE={rmse_hcm_speed})",
        f"STRIDE Estimated Speed (RMSE={rmse_stride_speed}) "
        f"alpha={np.round(alpha,2)}, "
        f"beta={np.round(beta,2)}, gamma="
        f"{np.round(gamma,2)}, "
        f"epsilon={np.round(epsilon,2)}, delta="
        f"{np.round(delta,2)}",
        f"STRIDE Estimated Speed (RMSE={rmse_calibrated_stride}) with "
        f"alpha={np.round(alpha_optimal,2)}, "
        f"beta={np.round(beta_optimal,2)}, gamma="
        f"{np.round(gamma_optimal,2)}, "
        f"epsilon={np.round(epsilon_optimal,2)}, delta="
        f"{np.round(delta_optimal,2)}, omega={np.round(omega_optimal, 2)}",
    ),
)
# Lazy way of plotting with plotly. Use express method to get the data then
# use a hack to get the data from express method and plot it. There should be
# cleaner and more intuitive way to generate these plots.
common_hover_fields = [
    "mainline_speed_limit",
    "FFS_lilian",
    "short_length_ls_ft",
    "Vrf_percent",
    "Vfr_percent",
    "num_of_mainline_lane_weaving_section",
    "mainline_grade",
    "interchange_density",
    "mainline_vol_pcu",
    "Vrf",
    "Vfr",
    "c_hcm_basic_seg",
    "bp_hcm_basic",
    "does_hcm_says_use_merge_diverge",
    "v_by_c",
    "is_v_by_c_over_1",
    "d_weave_hcm",
    "s_hcm_basic",
    "S_not_stride",
    "S_not_stride_with_unconstrained_calibrated_parameters",
    "s_weave_hcm",
]
plot_hcm = px.scatter(
    vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps,
    x="mainline_speed",
    y="s_weave_hcm",
    color="file_name",
    symbol="file_name",
    trendline="ols",
    hover_data=common_hover_fields,
)
# Add a 45 degree line to the data.
plot_hcm.add_trace(
    go.Scatter(
        x=[20, 95],
        y=[20, 95],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=False,
    )
)
plot_hcm_45_degree_line = plot_hcm
# Plotly plots have two components: data and layout. Extract the data.
data_plot_hcm = plot_hcm_45_degree_line["data"]
# Repeat above on STRIDE data. Again, this is the lazy approach; use functions or loops
# to have a cleaner implementation of this.
plot_stride = px.scatter(
    vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps,
    x="mainline_speed",
    y="S_not_stride",
    color="file_name",
    symbol="file_name",
    trendline="ols",
    hover_data=common_hover_fields,
)
plot_stride.add_trace(
    go.Scatter(
        x=[20, 95],
        y=[20, 95],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=False,
    )
)
plot_stride_45_degree_line = plot_stride
data_plot_stride = plot_stride_45_degree_line["data"]
# Repeat above on STRIDE data with calibrated parameter. Again, this is the lazy
# approach; use functions or loops to have a cleaner implementation of this.
# TODO: Add loop to make all this plotting stuff less clunky.
plot_stride_calibrated = px.scatter(
    vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps,
    x="mainline_speed",
    y="S_not_stride_with_unconstrained_calibrated_parameters",
    color="file_name",
    symbol="file_name",
    trendline="ols",
    hover_data=common_hover_fields,
)
plot_stride_calibrated.add_trace(
    go.Scatter(
        x=[20, 95],
        y=[20, 95],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=False,
    )
)
plot_stride_calibrated_45_degree_line = plot_stride_calibrated
data_plot_stride_calibrated = plot_stride_calibrated_45_degree_line["data"]

# Iterate through the hcm and stride data togather and generate scatters.
for dat1, dat2, dat3 in zip(
    data_plot_hcm, data_plot_stride, data_plot_stride_calibrated
):
    dat1["showlegend"] = False  # Remove duplicate legend. Plotly is weird!
    dat2["showlegend"] = False  # Remove duplicate legend. Plotly is weird!
    fig.add_trace(dat1, row=1, col=1)
    fig.add_trace(dat2, row=2, col=1)
    fig.add_trace(dat3, row=3, col=1)

# Make figures pretty.
fig.update_xaxes(
    title_text="Observed Speed (mph)", range=[20, 95], fixedrange=True, row=3, col=1
)
fig.update_yaxes(
    title_text="HCM Estimated Speed (mph)",
    range=[20, 95],
    fixedrange=True,
    row=1,
    col=1,
)
fig.update_yaxes(
    title_text="STRIDE Estimated Speed (mph)",
    range=[20, 95],
    fixedrange=True,
    row=2,
    col=1,
)
fig.update_yaxes(
    title_text="STRIDE Calibrated Parameters Estimated Speed (mph)",
    range=[20, 95],
    fixedrange=True,
    row=3,
    col=1,
)
fig.update_layout(autosize=True, height=1400, width=1300, margin=dict(l=350, t=20))
plot(
    fig,
    filename=os.path.join(path_figures_v1, "omega_HCM_vs_STRIDE_vs_Obs_speed_500_vph.html"),
    auto_open=True,
)

# Create plots for showing data points used for STRIDE method.
for file in keep_files:

    fig = go.Figure()

    vol_weave_df_simple_fil_stride_hcm_plot = vol_weave_df_simple.loc[
        lambda df: df.file_name == file
    ]
    fig.add_trace(
        go.Scatter(
            name="Uncongested volumes",
            mode="markers",
            x=vol_weave_df_simple_fil_stride_hcm_plot.mainline_vol,
            y=vol_weave_df_simple_fil_stride_hcm_plot.mainline_speed,
            marker=dict(color="blue", symbol="circle",),
        )
    )

    stride_df_plot = vol_weave_df_simple_fil_stride_hcm_extra_cols_hcm_steps.loc[
        lambda df: df.file_name == file
    ]
    fig.add_trace(
        go.Scatter(
            name="Data used for STRIDE",
            mode="markers",
            x=stride_df_plot.mainline_vol,
            y=stride_df_plot.mainline_speed,
            marker=dict(color="red", symbol="x",),
        ),
    )

    fig.update_layout(yaxis_range=[0, 100], xaxis_range=[0, 3000])

    plot(
        fig,
        filename=os.path.join(path_figures_v1, f"{file}_stride_fit_data.html"),
        auto_open=True,
    )
