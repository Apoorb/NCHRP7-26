"""
Script for simple merge sites:
    reading the uncongested volume and speed data along with metadata.
    Estimating HCM speed
    Estimating STRIDE speed with Nagui's parameter
    Estimating STRIDE parmaeters and then estimating speed.
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

# Output from get_ufl_merge_ramp_volumes_v2.py in the src-->data folder.
path_out_ufl_merge_ramp_upstream_volume_data = os.path.join(path_interim,
                                            "all_ufl_merge_ramp_upstream_mainline_data.xlsx")
all_ufl_merge_ramp_upstream_mainline_data = pd.read_excel(path_out_ufl_merge_ramp_upstream_volume_data)
# I used prebreakdown data, uncongested data, metadata for attributes, capacity data to
# create prebkdn_uncongested_weave_and_meta.csv in the clean_prebreakdown_data.py.
# clean_prebreakdown_data.py also handles duplicated pre-breakdown and uncongested volumes
path_volume_merge = os.path.join(path_interim, "prebkdn_uncongested_merge_meta.csv")

# Read the combined data for all merge sites and the metadata (output from the
# clean_prebreakdown_data.py).
vol_merge_df = pd.read_csv(path_volume_merge)
vol_merge_df_simple = (
    vol_merge_df
    .query("geometry_type == 'Simple merge'")
    .assign(Time=lambda df: pd.to_datetime(df.Time))
)
vol_merge_df_simple.file_name.unique()

vol_merge_df_simple_ramp_df = (
    vol_merge_df_simple
    .merge(
        all_ufl_merge_ramp_upstream_mainline_data,
        left_on=["file_name", "Time"],
        right_on=["site_nm", "tini"],
        how="inner" # inner join---keep rows where both mainline and ramp data is present.
    )
)

vol_merge_df_simple_ramp_df.file_name.unique()

vol_merge_df_simple_ramp_df_fil = (
    vol_merge_df_simple_ramp_df
    .sort_values(["file_name", "Time"])
    .rename(
        columns={
            "total_upstream_flow_rate": "v_F_upstream_vph",
            "upstream_flowrate_right_2nd_right_lane": "v_12_upstream_vph",
            "upstream_flowrate_outer_lanes_per_lane": "v_O_upstream_vphpl",
            "Num_outer_lanes": "n_o_upstream",
            "default_ramp_ffs_mph": "ffs_ramp_default",
            "ramp_flow_rate_per_lane": "v_R_vphpl"
        }
    )
    .assign(
        ET=lambda df: np.select(
            [df.mainline_grade <= 2, df.mainline_grade > 2], [2, 3]
        ),
        f_hv=lambda df: 1 / (1 + ((df.hv / 100) * (df.ET - 1))),
        mainline_vol_pcu=lambda df: df.mainline_vol / df.f_hv,
        v_R_pcphpl=lambda df: df.v_R_vphpl / df.f_hv,
        v_F_upstream_pcph=lambda df: df.v_F_upstream_vph / df.f_hv,
        v_O_upstream_pcphpl=lambda df: (df.v_O_upstream_vphpl / df.f_hv).fillna(0),
        v_12_upstream_pcph=lambda df: df.v_12_upstream_vph / df.f_hv,
    )
    .filter(
        items=[
            "file_name",
            "Time",
            "mainline_vol",
            "mainline_vol_pcu",
            "mainline_speed",
            "mainline_speed_limit",
            "ffs_cap_df",
            "length_of_acceleration_lane",
            "v_R_pcphpl",
            "number_of_lanes_upstream_from_ufl_detector_columns",
            "number_of_mainline_lane_downstream",
            "mainline_grade",
            "hv",
            "v_F_upstream_pcph",
            "v_12_upstream_pcph",
            "v_O_upstream_pcphpl",
            "n_o_upstream",
            "ffs_ramp_default",
        ]
    )
)
# QAQC dataframe
vol_merge_df_simple_ramp_df_fil_unique = vol_merge_df_simple_ramp_df_fil.drop_duplicates(
    "file_name"
)
# Get the required fields for applying STRIDE method
d_c_hcm_basic_seg = 45
a_hcm_basic_seg = 2
vol_merge_df_simple_ramp_df_fil_extra_cols = vol_merge_df_simple_ramp_df_fil.assign(
    c_hcm_basic_seg=lambda df: np.select(
        [
            (2200 + 10 * (df.ffs_cap_df - 50)) <= 2400,
            (2200 + 10 * (df.ffs_cap_df - 50)) > 2400,
        ],
        [2200 + 10 * (df.ffs_cap_df - 50), 2400],
    ),
    bp_hcm_basic=lambda df: 1000 + 40 * (75 - df.ffs_cap_df),
    s_hcm_basic=lambda df: np.select(
        [
            df.mainline_vol_pcu <= df.bp_hcm_basic,
            (df.mainline_vol_pcu > df.bp_hcm_basic)
            & (df.mainline_vol_pcu <= df.c_hcm_basic_seg),
            df.mainline_vol_pcu > df.c_hcm_basic_seg,
        ],
        [
            df.ffs_cap_df,
            df.ffs_cap_df
            - (
                (df.ffs_cap_df - (df.c_hcm_basic_seg / d_c_hcm_basic_seg))
                * ((df.mainline_vol_pcu - df.bp_hcm_basic) ** a_hcm_basic_seg)
                / ((df.c_hcm_basic_seg - df.bp_hcm_basic) ** a_hcm_basic_seg)
            ),
            np.nan,
        ],
    ),
).loc[
    lambda df: (df.mainline_speed >= 0.7 * df.ffs_cap_df)
    & (df.mainline_vol_pcu >= 500)
    # & (df.ffs_cap_df <= 75)
    & (~df.s_hcm_basic.isna())
]

# Compute HCM Predicted Speed
vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed=(
    vol_merge_df_simple_ramp_df_fil_extra_cols
    .assign(
        v12R=lambda df:df.v_12_upstream_pcph + df.v_R_pcphpl,
        M_s_hcm=lambda df: 0.321 + 0.0039 * np.exp(df.v12R / 1000)
                           - 0.002
                           * (df.length_of_acceleration_lane * df.ffs_ramp_default / 1000),
        S_R_hcm=lambda df: df.ffs_cap_df - (df.ffs_cap_df - 42) * df.M_s_hcm,
        S_O_hcm=lambda df: np.select(
            [df.v_O_upstream_pcphpl < 500, df.v_O_upstream_pcphpl <= 2300, df.v_O_upstream_pcphpl > 2300],
            [df.ffs_cap_df, df.ffs_cap_df - 0.0036 * (df.v_O_upstream_pcphpl-500), df.ffs_cap_df - 6.53 - 0.006*(df.v_O_upstream_pcphpl-2300)]
        ),
        S_hcm=lambda df: (df.v12R + df.v_O_upstream_pcphpl * df.n_o_upstream)
                         / ((df.v12R / df.S_R_hcm) + (df.v_O_upstream_pcphpl * df.n_o_upstream / df.S_O_hcm))
    )
)
rmse_hcm_speed = math.sqrt(
    mean_squared_error(
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.mainline_speed,
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.S_hcm,
    )
)
rmse_hcm_speed = np.round(rmse_hcm_speed, 2)
print(f"Root mean squared error for HCM method = {rmse_hcm_speed}")


# Compute S_knot from STRIDE method with Nagui's calibrated parameters
# STRIDE calibrated parameters from Nagui's research
alpha = 0.025
beta = 17.302
gamma = 0.344
delta = 0.369
epsilon = 3
omega = 1

def get_S_not_STRIDE(row, alpha, beta, gamma, epsilon, delta, omega=1):
    Vrf = row.v_R_pcphpl
    Vfr = 0
    nl = row.number_of_mainline_lane_downstream
    mainline_vol_pcu = row.mainline_vol_pcu
    short_length_ls_ft = row.length_of_acceleration_lane
    s_hcm_basic = row.s_hcm_basic
    return (
        s_hcm_basic
        - alpha
        * (((beta * Vrf + omega * Vfr) / (nl) ** epsilon) ** gamma)
        * (mainline_vol_pcu - 500)
        * (1 / short_length_ls_ft) ** delta
    )


vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.loc[
    :, "S_not_stride"
] = vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.apply(
    get_S_not_STRIDE,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    epsilon=epsilon,
    delta=delta,
    omega=omega,
    axis=1,
)
# QAQC dataframe
test_stride_speed_calc = vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.drop_duplicates(
    "file_name"
)

# Get RMSE
rmse_stride_speed = math.sqrt(
    mean_squared_error(
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.mainline_speed,
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.S_not_stride,
    )
)
rmse_stride_speed = np.round(rmse_stride_speed, 2)
print(f"Root mean squared error for STRIDE method = {rmse_stride_speed}")
# Filter to columns that are needed for the STRIDE equation.
stride_model_fit_df = (
    vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.assign(
        s_knot_minus_s_b=lambda df: df.mainline_speed - df.s_hcm_basic,
        Vfr=0
    )
    .rename(columns={"number_of_mainline_lane_downstream": "nl"})
    .filter(  # The order on the variables below matter. They are being read by
        # curve_fit_stride in this order. Do not change the variable order below!
        items=[
            "s_knot_minus_s_b",
            "v_R_pcphpl",
            "Vfr",
            "nl",
            "mainline_vol_pcu",
            "length_of_acceleration_lane",
        ]
    )
)

# Estimate the STRIDE parameters---unconstrained optimization
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

# Use the optimal values from weave
alpha_optimal = 0.12
beta_optimal = 0.44
gamma_optimal = 0.63
epsilon_optimal = 3.23
delta_optimal = 0.67
omega_optimal = 1.19

vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.loc[
    :, "S_not_stride_with_unconstrained_calibrated_parameters"
] = vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.apply(
    get_S_not_STRIDE,
    alpha=alpha_optimal,
    beta=beta_optimal,
    gamma=gamma_optimal,
    epsilon=epsilon_optimal,
    delta=delta_optimal,
    omega=omega_optimal,
    axis=1,
)
rmse_calibrated_stride = math.sqrt(
    mean_squared_error(
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.mainline_speed,
        vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.S_not_stride_with_unconstrained_calibrated_parameters,
    )
)
rmse_calibrated_stride = np.round(rmse_calibrated_stride, 2)
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
    "ffs_cap_df",
    "length_of_acceleration_lane",
    "v_R_pcphpl",
    "number_of_mainline_lane_downstream",
    "mainline_grade",
    "mainline_vol_pcu",
    "c_hcm_basic_seg",
    "bp_hcm_basic",
    "s_hcm_basic",
    "S_not_stride",
    "S_not_stride_with_unconstrained_calibrated_parameters",
    "S_hcm",
]
plot_hcm = px.scatter(
    vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed,
    x="mainline_speed",
    y="S_hcm",
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
    vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed,
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
    vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed,
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
    filename=os.path.join(path_figures_v1, "merge_omega_HCM_vs_STRIDE_vs_Obs_speed_500_vph.html"),
    auto_open=True,
)

# Create plots for showing data points used for STRIDE method.
for file in vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.file_name.unique():

    fig = go.Figure()

    vol_weave_df_simple_fil_stride_hcm_plot = vol_merge_df_simple_ramp_df_fil.loc[
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

    stride_df_plot = vol_merge_df_simple_ramp_df_fil_extra_cols_hcm_merge_speed.loc[
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
        filename=os.path.join(path_figures_v1, f"merge_{file}_stride_fit_data.html"),
        auto_open=True,
    )
