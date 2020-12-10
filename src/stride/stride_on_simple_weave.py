import os
import pandas as pd
from src.utils import get_project_root
import plotly.io as pio
import numpy as np

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
path_volume_weave = os.path.join(path_interim, "prebkdn_uncongested_weave_and_meta.csv")
path_stride_streetlight_df = os.path.join(path_raw, "STRIDE Weaving Capacity.xlsx")

vfr_rf_df = (
    pd.read_excel(path_stride_streetlight_df, skiprows=1, nrows=13)
    .rename(
        columns={"Unnamed: 0": "file_name", "Vrf": "Vrf_percent", "Vfr": "Vfr_percent"}
    )
    .assign(file_name=lambda df: df.file_name.str.strip().str.title())
    .filter(items=["file_name", "Vrf_percent", "Vfr_percent"])
)
vfr_rf_df.file_name
vol_weave_df = pd.read_csv(path_volume_weave)
vol_weave_df_simple = vol_weave_df.query("geometry_type == 'Simple ramp weave'")
vol_weave_df_simple.file_name.unique()
keep_files = [
    "50_Simple Ramp Weave_1",
    "51_Simple Ramp Weave_2",
    "52_Simple Ramp Weave_3",
    "53_Simple Ramp Weave_4",
    "56_Simple Ramp Weave_7",
    "57_Simple Ramp Weave_8",
    "59_Simple Ramp Weave_10",
    "61_Simple Ramp Weave_12",
    "62_Simple Ramp Weave_13",
    "63_Simple Ramp Weave_14",
    "64_Simple Ramp Weave_15",
]
vol_weave_df_simple_fil = vol_weave_df_simple.loc[
    lambda df: df.file_name.isin(keep_files)
]

set(vfr_rf_df.file_name) - set(keep_files)
set(keep_files) - set(vfr_rf_df.file_name)

vol_weave_df_simple_fil_stride = (
    vol_weave_df_simple_fil.merge(vfr_rf_df, on="file_name", how="left")
    .filter(
        items=[
            "file_name",
            "mainline_vol",
            "mainline_speed",
            "mainline_speed_limit",
            "ffs_cap_df",
            "short_length_ls_ft",
            "Vrf_percent",
            "Vfr_percent",
            "num_of_mainline_lane_weaving_section",
            "mainline_grade",
            "hv",
        ]
    )
    .assign(
        ET=lambda df: np.select(
            [df.mainline_grade <= 2, df.mainline_grade > 2], [2, 3]
        ),
        f_hv=lambda df: 1 / (1 + ((df.hv / 100) * (df.ET - 1))),
        mainline_vol_pcu=lambda df: df.mainline_vol / df.f_hv,
    )
)

vol_weave_df_simple_fil_stride_unique = vol_weave_df_simple_fil_stride.drop_duplicates(
    "file_name"
)

d_c_hcm_basic_seg = 45
a_hcm_basic_seg = 2
vol_weave_df_simple_fil_stride_fil = vol_weave_df_simple_fil_stride.assign(
    ffs_sl_max=lambda df: df[["ffs_cap_df", "mainline_speed_limit"]].max(axis=1),
    Vrf=lambda df: df.Vrf_percent
    * df.mainline_vol_pcu
    * df.num_of_mainline_lane_weaving_section,
    Vfr=lambda df: df.Vfr_percent
    * df.mainline_vol_pcu
    * df.num_of_mainline_lane_weaving_section,
    c_hcm_basic_seg=lambda df: np.select(
        [
            (2200 + 10 * (df.ffs_sl_max - 50)) <= 2400,
            (2200 + 10 * (df.ffs_sl_max - 50)) > 2400,
        ],
        [2200 + 10 * (df.ffs_sl_max - 50), 2400],
    ),
    bp_hcm_basic=lambda df: 1000 + 40 * (75 - df.ffs_sl_max),
    s_hcm_basic=lambda df: np.select(
        [
            df.mainline_vol_pcu <= df.bp_hcm_basic,
            (df.mainline_vol_pcu > df.bp_hcm_basic)
            & (df.mainline_vol_pcu <= df.c_hcm_basic_seg),
            df.mainline_vol_pcu > df.c_hcm_basic_seg,
        ],
        [
            df.ffs_sl_max,
            df.ffs_sl_max
            - (
                (df.ffs_sl_max - (df.c_hcm_basic_seg / d_c_hcm_basic_seg))
                * ((df.mainline_vol_pcu - df.bp_hcm_basic) ** a_hcm_basic_seg)
                / ((df.c_hcm_basic_seg - df.bp_hcm_basic) ** a_hcm_basic_seg)
            ),
            np.nan,
        ],
    ),
).loc[
    lambda df: (df.mainline_speed >= 0.75 * df.ffs_sl_max)
    & (df.mainline_vol_pcu >= 1000)
    & (df.ffs_sl_max <= 75)
    & (~df.s_hcm_basic.isna())
]


test = vol_weave_df_simple_fil_stride_fil.drop_duplicates("file_name")

stride_model_fit_df = (
    vol_weave_df_simple_fil_stride_fil.assign(
        s_knot_minus_s_b=lambda df: df.mainline_speed - df.s_hcm_basic
    )
    .rename(columns={"num_of_mainline_lane_weaving_section": "nl"})
    .filter(
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

from scipy.optimize import curve_fit


def curve_fit_stride(X, alpha, beta, gamma, epsilon, delta):
    s_knot_minus_s_b, Vrf, Vfr, nl, mainline_vol_pcu, short_length_ls_ft = X
    return (
        s_knot_minus_s_b
        + alpha
        * (((beta * Vrf + Vfr) / (nl) ** epsilon) ** gamma)
        * (mainline_vol_pcu - 500)
        * (1 / short_length_ls_ft) ** delta
    )


Y = np.transpose(np.zeros(len(stride_model_fit_df)))
X = np.array(stride_model_fit_df.values.T)
len(curve_fit_stride(X, 0.025, 17.302, 0.344, 3, 0.369))
np.set_printoptions(suppress=True)
initial_guess = [1, 1, 1, 1, 1]
popt, pcov = curve_fit(curve_fit_stride, X, Y, p0=initial_guess, maxfev=5000)
alpha, beta, gamma, epsilon, delta = popt
Error = curve_fit_stride(X, alpha, beta, gamma, epsilon, delta)

def curve_fit_stride_test(X, alpha, beta, gamma, epsilon, delta):
    s_knot_minus_s_b, Vrf, Vfr, nl, mainline_vol_pcu, short_length_ls_ft = X
    return (
        + alpha
        * (((beta * Vrf + Vfr) / (nl) ** epsilon) ** gamma)
        * (mainline_vol_pcu - 500)
        * (1 / short_length_ls_ft) ** delta
    )

# QAQC
curve_fit_stride_test(X, alpha, beta, gamma, epsilon, delta)
x1 = X[:, 0]
((beta * x1[1] + x1[2]) / x1[3]**epsilon)**gamma
