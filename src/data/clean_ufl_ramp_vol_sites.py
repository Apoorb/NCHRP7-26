import os
import pandas as pd
import numpy as np
import inflection
import re
from sklearn import tree
from sklearn import preprocessing
import graphviz
import seaborn as sns
from src.utils import get_project_root
from src.data import clean_prebreakdown_data

# Set paths:
path_to_project = str(get_project_root())
path_meta = (
    r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - "
    r"NCHRP 07-26\Task 6 - Execute Phase II Plan\Site Analysis & Metadata"
)
path_site_char = os.path.join(path_meta, "NCHRP 07-26_Master_Database_shared.xlsx")
path_prebreakdown_ufl = os.path.join(
    r"C:\Users\abibeka\Documents_axb\nchrp7-26\regression_analysis_prebreakdown",
    "pre_breakdown_analysis_v1.xlsx",
)
path_interim = os.path.join(path_to_project, "data", "interim")
path_figures = r"/figures"


def get_prebreakdown_data_ufl(path_prebreakdown_ufl_):
    x1 = pd.ExcelFile(path_prebreakdown_ufl)
    sheet_names = x1.sheet_names
    list_prebreakdown_df = []
    for sheet_nm in x1.sheet_names:
        if sheet_nm == "pivot_tables":
            continue
        temp_df = x1.parse(sheet_nm).assign(file=sheet_nm.strip())
        temp_df = temp_df.rename(
            columns={col: inflection.underscore(col) for col in temp_df.columns}
        )
        temp_df = (
            temp_df
            .rename(
                columns={
                    "mainline_vol": "prebreakdown_vol",
                    "mainline_speed": "prebreakdown_speed",
                }
            )
        )
        list_prebreakdown_df.append(temp_df)
    return pd.concat(list_prebreakdown_df)


if __name__ == "__main__":
    site_sum_merge = clean_prebreakdown_data.read_site_data(
        path_=path_site_char, sheet_name_="Merge"
    )
    site_sum_merge_fil = clean_prebreakdown_data.clean_site_sum_merge(
        site_sum_merge_=site_sum_merge, iloc_max_row=36
    )
    prebreakdown_df_ufl = get_prebreakdown_data_ufl(path_prebreakdown_ufl)
    prebreakdown_df_ufl_meta = prebreakdown_df_ufl.merge(
        site_sum_merge_fil, on="file", how="left"
    )
    prebreakdown_df_ufl_meta.to_csv(
        os.path.join(path_interim, "prebreakdown_ufl_merge_and_meta.csv"), index=False
    )

