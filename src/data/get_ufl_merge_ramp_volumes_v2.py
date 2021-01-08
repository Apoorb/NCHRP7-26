"""
Process UFL ramp data and associate with merge site names.
"""
from src.utils import get_project_root
import os
import pandas as pd
import glob
import re

path_to_project = str(get_project_root())
path_raw = os.path.join(path_to_project, "data", "raw")
path_to_ramp_site_nm = os.path.join(path_raw, "merge_mainline_to_ramp_detectors.txt")
path_to_ufl_ramp_data_pat = os.path.join(path_raw, "ufl_ramp_data", "*.xlsx")
path_to_ufl_ramp_data_files = glob.glob(path_to_ufl_ramp_data_pat)
path_to_interim = os.path.join(path_to_project, "data", "interim")
path_out_ufl_merge_ramp_upstream_mainline_data = os.path.join(path_to_interim,
                                            "all_ufl_merge_ramp_upstream_mainline_data.xlsx")
re_ramp_number = re.compile("\d+")
# Dictionary with ramp detector number and file number
det_ufl_dict = {
    int(re.search(re_ramp_number, os.path.basename(file)).group()): file
    for file in path_to_ufl_ramp_data_files
}
ramp_site_nm_mapper = pd.read_csv(
    path_to_ramp_site_nm, comment="#", sep=";", thousands=","
)
ramp_site_nm_mapper.columns = ["site_nm", "mainline_det", "ramp_det","mainline_upstream_det"]
list_ufl_ramp_mainline_upstream_dfs = []
for no, row in ramp_site_nm_mapper.iterrows():
    site_nm = row.site_nm
    ramp_det = row.ramp_det
    mainline_upstream_det = row.mainline_upstream_det
    ufl_ramp_df = pd.read_excel(det_ufl_dict[ramp_det])
    ufl_upstream_mainline_df = pd.read_excel(det_ufl_dict[mainline_upstream_det])
    # Process ramp data.
    ######################################################################################
    re_flowrate_cols = re.compile(r"L(\d)hourFlow", flags=re.I)
    # Get the columns that have the flow are by lane. Will use the data for veh/hr/ln.
    keep_cols = [col for col in ufl_ramp_df.columns if re.search(re_flowrate_cols, col)]
    # Check if have data from all lanes.
    try:
        assert all(ufl_ramp_df[keep_cols].sum(axis=1) == ufl_ramp_df.Flow_vph), (
            "Didn't catch all volume columns.")
    except AssertionError as aserr:
        print(aserr)

    if (ramp_det == 10109710) & (site_nm == '13_Simple Merge_13'):
        keep_cols.remove('L1hourFlow')  # Remove the Lane 1 flow-rates as they are close
                                        # to 0 for most time period.
        #DEBUG:
        ufl_ramp_df.L1hourFlow.describe()

    ufl_ramp_df_fil = (
        ufl_ramp_df
        .filter(items=keep_cols+["tini"])
        .assign(
            tini=lambda df: pd.to_datetime(df.tini),
            ramp_flow_rate_per_lane=lambda df: df[keep_cols].mean(axis=1),
            site_nm=site_nm,
            ramp_det_no=ramp_det,
        )
        .loc[lambda df: df.ramp_flow_rate_per_lane > 0]
        .filter(items=["site_nm", "ramp_det_no", "tini", "ramp_flow_rate_per_lane"])
    )

    # Process upstream mainline detector data.
    ######################################################################################
    re_flowrate_1_2_cols = re.compile(r"L(\d)hourFlow", flags=re.I)
    # Get the columns that have the flow are by lane. Will use the data for veh/hr/ln.
    import collections
    lane_no_col_name = {int(re.search(re_flowrate_1_2_cols, col).group(1)): col
                 for col in ufl_upstream_mainline_df.columns
                 if re.search(re_flowrate_1_2_cols, col)}
    import collections
    lane_no_col_name_sorted = collections.OrderedDict(sorted(lane_no_col_name.items(), reverse=True))
    # Get the right 2 lane (Lanes with Max and Max -1 lane number. In can of 4 lanes we
    # are looking for lane 4 and 3.
    lane_iter = iter(lane_no_col_name_sorted)
    right_lane = next(lane_iter)
    second_lane_from_right = next(lane_iter)
    right_2_lane_cols = [col_desc
     for lane_no, col_desc in lane_no_col_name.items()
     if lane_no in [right_lane, second_lane_from_right]]
    number_of_outer_lanes = len(lane_no_col_name) - 2
    number_of_lanes_upstream = len(lane_no_col_name)
    # Flow Rate for Right 2 lanes
    ufl_upstream_mainline_df_fil = (
        ufl_upstream_mainline_df
        .assign(
            upstream_flowrate_right_2nd_right_lane=(
                lambda df: df[right_2_lane_cols].sum(axis=1)),
            Num_outer_lanes=number_of_outer_lanes,
            number_of_lanes_upstream_from_ufl_detector_columns=number_of_lanes_upstream,
            total_upstream_flow_rate=lambda df: (
                df[lane_no_col_name.values()].sum(axis=1)),
            upstream_flowrate_outer_lanes_per_lane=(
                lambda df: (df.total_upstream_flow_rate
                            - df.upstream_flowrate_right_2nd_right_lane)
                            / number_of_outer_lanes
            ),
            default_ramp_ffs_mph=35,
            site_nm=site_nm,
            mainline_upstream_det=mainline_upstream_det,
        )
        .filter(items=["site_nm", "mainline_upstream_det", "tini",
                       "Num_outer_lanes",
                       "number_of_lanes_upstream_from_ufl_detector_columns",
                       "default_ramp_ffs_mph",
                       "total_upstream_flow_rate",
                       "upstream_flowrate_right_2nd_right_lane",
                       "upstream_flowrate_outer_lanes_per_lane"])
        .loc[lambda df: df.upstream_flowrate_right_2nd_right_lane > 0]
    )

    # Merge ramp and upstream mainline detector data.
    ######################################################################################
    ufl_upstream_mainline_df_fil_ramp_df = ufl_upstream_mainline_df_fil.merge(
        ufl_ramp_df_fil,
        on=["site_nm", "tini"],
        how="inner"
    )
    list_ufl_ramp_mainline_upstream_dfs.append(ufl_upstream_mainline_df_fil_ramp_df)


all_ufl_ramp_upstream_mainline_df = pd.concat(
    list_ufl_ramp_mainline_upstream_dfs
)

all_ufl_ramp_upstream_mainline_df.to_excel(path_out_ufl_merge_ramp_upstream_mainline_data)
