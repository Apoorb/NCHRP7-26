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
path_out_ufl_merge_ramp_data = os.path.join(path_to_interim,
                                            "all_ufl_merge_ramp_data.xlsx")
re_ramp_number = re.compile("\d+")
# Dictionary with ramp detector number and file number
ramp_det_ufl_dict = {
    int(re.search(re_ramp_number, os.path.basename(file)).group()): file
    for file in path_to_ufl_ramp_data_files
}
ramp_site_nm_mapper = pd.read_csv(
    path_to_ramp_site_nm, comment="#", sep=";", thousands=","
)
ramp_site_nm_mapper.columns = ["site_nm", "mainline_det", "ramp_det"]
list_ufl_ramp_dfs = []
for no, row in ramp_site_nm_mapper.iterrows():
    site_nm = row.site_nm
    ramp_det = row.ramp_det
    ufl_ramp_df = pd.read_excel(ramp_det_ufl_dict[ramp_det])
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
    list_ufl_ramp_dfs.append(ufl_ramp_df_fil)


all_ufl_ramp_df = pd.concat(
    list_ufl_ramp_dfs
)

all_ufl_ramp_df.to_excel(path_out_ufl_merge_ramp_data)
