import os
import pandas as pd
import re

path_to_interim = r"/data/interim"

path_to_data_files = (
    r"C:\Users\abibeka\Github\NCHRP 7-26\data\interim\pre_breakdown_data"
)

df_list = []
for file in os.listdir(path_to_data_files):
    df = pd.read_csv(os.path.join(path_to_data_files, file))
    if len(df) == 0:
        df.loc[0, "filename"] = file
    df.loc[:, "filename"] = file
    df_list.append(df)

df_concat = pd.concat(df_list)
df_concat_grp = (
    df_concat.groupby("filename")[["MainlineVol", "MainlineSpeed"]]
    .describe()
    .reset_index()
)
geometry_name = re.compile(
    "^(?P<geometry_type>[0-9]{1,3}_.*_[0-9]{1,2}_?\w*?)_pre_brkdn.csv$"
)
df_concat_grp["geometry_type"] = df_concat_grp.filename.str.extract(
    geometry_name
).values

df_concat_grp.to_excel(
    os.path.join(path_to_interim, "summary_statistics_based_on_Azy's code.xlsx")
)
