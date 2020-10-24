# Developed by Azy - aavr@kittelson.com

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
import os.path
from scipy.stats import linregress
import warnings
import xlsxwriter
from openpyxl import load_workbook


def define_defaults(file_, time_threshold_):
    # root = tk.Tk()
    # root.withdraw()
    # print("Choose the mainline file....")
    # file_path = filedialog.askopenfilename()
    # print(file_path)
    file_path = file_
    start = datetime.datetime.utcnow()
    # raw_data = pd.read_csv("time_thresholds.csv", header=None)
    # Mat = np.matrix(raw_data)
    # time = Mat[:, 0]
    # time = np.squeeze(np.asarray(time))

    time = [0, 0]
    time[0] = datetime.datetime.strptime(time_threshold_[0], "%H:%M:%S")
    time[1] = datetime.datetime.strptime(time_threshold_[1], "%H:%M:%S")
    # print(time[0].time())
    print("Filtered Start Time: " + str(time[0].time()))
    print("Filtered End Time: " + str(time[1].time()))

    return (file_path, start, time)


def read_file_data(file_path, time, is_flow_volume, remove_lane_data_=[]):
    raw_data = pd.read_csv(file_path, header=0)
    if is_flow_volume == "flow":
        raw_data.Volume = raw_data.Volume / 4
    if sum(raw_data.duplicated(["Timestamp", "Lane"], keep=False)) != 0:
        duplicated_data = raw_data[
            raw_data.duplicated(["Timestamp", "Lane"], keep=False)
        ]
        raw_data = raw_data[~raw_data.duplicated(["Timestamp", "Lane"], keep="first")]
    if len(remove_lane_data_) != 0:
        raw_data = raw_data.loc[lambda df: ~df.Lane.isin(remove_lane_data_)]
    num_lanes = len(raw_data.Lane.unique())
    raw_data.Timestamp = pd.to_datetime(raw_data.Timestamp)
    raw_data_mainline_pivot = raw_data.pivot(
        index="Timestamp", columns=["Lane"], values=["Volume", "Speed"]
    )
    raw_data_mainline_pivot_fil = raw_data_mainline_pivot.between_time(
        start_time=time[0].time(), end_time=time[1].time()
    )
    volumes = raw_data_mainline_pivot_fil["Volume"].values.T.tolist()
    speeds = raw_data_mainline_pivot_fil["Speed"].values.T.tolist()
    measurement_duration = list(raw_data_mainline_pivot_fil.index.to_pydatetime())
    return speeds, volumes, measurement_duration, num_lanes


def read_ramp_data(file_path, time, is_flow_volume):
    raw_data = pd.read_csv(file_path, header=0)
    if is_flow_volume == "flow":
        raw_data.Volume = raw_data.Volume / 4
    if sum(raw_data.duplicated(["Timestamp", "Lane"], keep=False)) != 0:
        duplicated_data = raw_data[
            raw_data.duplicated(["Timestamp", "Lane"], keep=False)
        ]
        raw_data = raw_data[~raw_data.duplicated(["Timestamp", "Lane"], keep="first")]
    raw_data_ramp = raw_data.groupby("Timestamp").Volume.mean()
    raw_data_ramp.index = pd.to_datetime(raw_data_ramp.index)
    volumes = raw_data_ramp.values.T.tolist()
    measurement_duration = list(raw_data_ramp.index.to_pydatetime())

    for i in range(len(measurement_duration)):
        measurement_duration[i] = str(
            datetime.datetime.strptime(
                str(measurement_duration[i]), "%Y-%m-%d %H:%M:%S"
            )
        )

    return volumes, measurement_duration

    root = tk.Tk()
    root.withdraw()
    print("Choose the Ramp file....")
    file_path = filedialog.askopenfilename()
    # print(file_path)
    no_of_lanes = str(file_path).split("_0_")[1].split(".xls")[0]
    # print(no_of_lanes)
    df = pd.read_excel(file_path)

    if no_of_lanes == "1":
        l1 = df["L1Volume"].tolist()
        time = df["tini"].tolist()
        # print(time[0].day)
    elif no_of_lanes == "2":
        l1 = (df["L1Volume"] + df["L2Volume"]).tolist()
        time = df["tini"].tolist()

    elif no_of_lanes == "3":
        l1 = (df["L1Volume"] + df["L2Volume"] + df["L3Volume"]).tolist()
        time = df["tini"].tolist()
    else:
        l1 = []
        time = []

    return (l1, time)


def approach_sped_volume_ramp(
    speed_1,
    volume_1,
    date_time,
    lanes,
    ramp_volume,
    ramp_time,
    add_ramp_volume_to_mainline,
):
    # print(len(speed_1))
    # print(len(volume_1[0]))
    # print(lanes)
    approach_vol = []
    approach_speed = []
    approach_time = []
    for i in range(len(speed_1[0])):
        nr = 0
        dr = 0
        volume = 0
        temp = 0
        for j in range(len(speed_1)):

            if (volume_1[j][i] != 0) & (not np.isnan(volume_1[j][i])):
                nr = nr + (speed_1[j][i] * volume_1[j][i])
                dr = dr + volume_1[j][i]
                volume = volume + volume_1[j][i]
                temp = 1
            else:
                temp = 0
                break

        if temp == 1:
            approach_time.append(str(date_time[i]))
            approach_speed.append(int(nr / dr))
            approach_vol.append(int(volume / len(volume_1)))

    # print((approach_speed))
    # print((approach_vol))
    # print((approach_time))
    # print(ramp_time)
    ramp_vol = []
    for i in range(len(approach_time)):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            idx = np.where(np.array(ramp_time) == approach_time[i])[0]
        # print(idx)
        if len(idx) != 0:
            idx = idx[0]
            ramp_vol.append(ramp_volume[idx])
            if add_ramp_volume_to_mainline:
                approach_vol[i] = approach_vol[i] + ramp_volume[idx]
            # print(ramp_volume[idx]*4)
        else:
            # print(0)
            ramp_vol.append(np.nan)

    return (approach_vol, approach_speed, ramp_vol, approach_time)


def FFS_fn(volume, speed, x):
    FFS = 0
    temp = 0
    for i in range(len(volume)):
        # print(volume[i])
        if float(volume[i]) * x <= 500 and float(volume[i]) * x > 0:
            FFS = FFS + float(speed[i])
            temp = temp + 1
    return FFS / temp


def roundup(x):
    seq = int(math.ceil(x / 100.0))
    round_100 = seq * 100
    return (seq + 1, round_100)


def breakdown_fn(speed, volume, ramp_volume, FFS, x):
    breakdown_speed = []
    breakdown_volume = []
    breakdown_index = []
    uncong_index = []
    uncong_speed = []
    uncong_volume = []
    ramp_bd_vol = []
    ramp_uncong_vol = []
    volume[0] = float(volume[0]) * x
    ramp_volume[0] = float(ramp_volume[0]) * x
    speed[0] = float(speed[0])
    uncong_index.append(0)
    uncong_speed.append(speed[0])
    uncong_volume.append((volume[0]))
    for i in range(1, len(speed)):
        speed[i] = float(speed[i])
        volume[i] = float(volume[i]) * x
        ramp_volume[i] = float(ramp_volume[i]) * x
        if float(speed[i]) <= 0.75 * FFS:
            breakdown_index.append(i)
            breakdown_speed.append(speed[i])
            breakdown_volume.append((volume[i]))
            ramp_bd_vol.append(ramp_volume[i])
        else:
            uncong_index.append(i)
            uncong_speed.append(speed[i])
            uncong_volume.append((volume[i]))
            ramp_uncong_vol.append(ramp_volume[i])

    seq, round_100 = roundup(max(uncong_volume))
    flow_seq = np.linspace(1000, round_100, seq - 10)
    # print(flow_seq)
    # flow_seq = np.linspace(0, 1400, 15)
    return (
        breakdown_index,
        breakdown_volume,
        breakdown_speed,
        uncong_index,
        uncong_speed,
        uncong_volume,
        flow_seq,
        seq,
        volume,
        speed,
        ramp_uncong_vol,
        ramp_bd_vol,
    )


def pre_post_breakdown_index_fn(breakdown_index, x):
    if x == 0:
        pre_breakdown_index = []
        post_breakdown_index = []
        pre_breakdown_index.append(breakdown_index[0] - 1)
        for i in range(1, len(breakdown_index)):
            if float(breakdown_index[i]) - float(breakdown_index[i - 1]) > 1:
                post_breakdown_index.append(breakdown_index[i - 1] + 1)
                pre_breakdown_index.append(breakdown_index[i] - 1)
        post_breakdown_index.append(breakdown_index[len(breakdown_index) - 1] + 1)
        return (pre_breakdown_index, post_breakdown_index)
    else:
        breakdown_index_1 = []
        breakdown_index_1.append(breakdown_index[0])
        for i in range(1, len(breakdown_index)):
            if float(breakdown_index[i]) - float(breakdown_index[i - 1]) > 1:
                breakdown_index_1.append(breakdown_index[i])
        # print(breakdown_index_1)
        return breakdown_index_1


def print_pre_BD_data(
    speed, volume, ramp_volume_1, approach_time, pre_breakdown_index, out_file_path
):
    pre_bd_vol = []
    pre_bd_speed = []
    pre_bd_ramp_vol = []
    pre_bd_time = []
    for pre_bd in pre_breakdown_index:

        if volume[pre_bd] >= 1000:
            pre_bd_vol.append(volume[pre_bd])
            pre_bd_speed.append(speed[pre_bd])
            pre_bd_ramp_vol.append(ramp_volume_1[pre_bd])
            pre_bd_time.append(approach_time[pre_bd])

    df_content = {
        "Time": pre_bd_time,
        "MainlineSpeed": pre_bd_speed,
        "MainlineVol": pre_bd_vol,
    }
    df = pd.DataFrame(df_content)
    df.to_csv(out_file_path, index=False)


def get_site_summary_stats(path_to_site_summary_):
    site_df = pd.read_excel(path_to_site_summary_, sheet_name="Site Summary",)
    site_df_fil_ = (
        site_df.rename(
            columns={
                "File Name New": "file_nm",
                "Fix FFS": "fix_ffs",
                " FFS": "ffs",
                "Sl.No": "sl_no",
            }
        )
        .filter(items=["sl_no", "file_nm", "fix_ffs", "ffs"], axis=1)
        .loc[lambda x: ~x.sl_no.isna()]
        .assign(
            fix_ffs=lambda df: np.select(
                [df.fix_ffs.str.strip().str.upper() == "Y", df.fix_ffs.isna()],
                [1, 0],
                "Error",
            ),
            file_nm=lambda df: df.file_nm + ".csv",
            time_thresholds=lambda df: np.select(
                [
                    df.file_nm.isin(
                        [
                            "119_Ramp Metered_1_Before.csv",
                            "120_Ramp Metered_1_After.csv",
                            "121_Ramp Metered_2_Before.csv",
                            "122_Ramp Metered_2_After.csv",
                            "125_Ramp Metered_4_Before.csv",
                            "126_Ramp Metered_4_After.csv",
                        ],
                    ),
                    df.file_nm.isin(
                        [
                            "123_Ramp Metered_3_Before.csv",
                            "124_Ramp Metered_3_After.csv",
                        ]
                    ),
                ],
                ["05:00:00-10:00:00", "12:00:00-20:00:00",],
                "00:00:00-23:45:00",
            ),
        )
    )
    return site_df_fil_


def get_file_path_dict(path_to_kai_clean_data, path_to_uw_clean_data):
    files_dict_ = {
        file: path_to_kai_clean_data
        for file in os.listdir(path_to_kai_clean_data)
        if file != "weave_ramp_volumes"
    }
    files_dict_.update(
        {file: path_to_uw_clean_data for file in os.listdir(path_to_uw_clean_data)}
    )
    return files_dict_


def get_weave_ramp_file_dict(path_to_weave_ramp_data_):
    ramp_files = os.listdir(path_to_weave_ramp_data_)
    ramp_files_dict_ = {
        file.split("_")[0]: os.path.join(path_to_weave_ramp_data_, file)
        for file in ramp_files
    }
    return ramp_files_dict_


if __name__ == "__main__":
    path_to_site_summary = (
        r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme -"
        r" NCHRP 07-26\Task 6 - Execute Phase II Plan\Site Analysis"
        r" & Metadata\NCHRP07-26_Site_summary_shared.xlsx"
    )
    path_to_kai_clean_data = (
        r"C:\Users\abibeka\Documents_axb\nchrp7-26\Data\05_organized_KAI_Data"
    )
    path_to_uw_clean_data = (
        r"C:\Users\abibeka\Documents_axb\nchrp7-26\Data\04_organized_UW_Data"
    )
    path_to_weave_ramp_data = (
        r"C:\Users\abibeka\Documents_axb\nchrp7-26\Data\05_organized_KAI_Data"
        r"\weave_ramp_volumes"
    )
    path_to_out_files = (
        r"C:\Users\abibeka\Github\NCHRP 7-26\data\interim\pre_breakdown_data"
    )
    site_df_fil = get_site_summary_stats(path_to_site_summary)
    files_dict = get_file_path_dict(path_to_kai_clean_data, path_to_uw_clean_data)
    ramp_files_dict = get_weave_ramp_file_dict(path_to_weave_ramp_data)
    # Correct issue with KAI volumes being volumes and UW volumes being flow rates.
    is_flow_volume_dict = {
        file: "volume" for file in os.listdir(path_to_kai_clean_data)
    }
    is_flow_volume_dict.update(
        {file: "flow" for file in os.listdir(path_to_uw_clean_data)}
    )

    print("Do you want to re-run previously ran file: (y/n)")
    user_input = input()
    if user_input == "y":
        ran_files_no = []
    else:
        ran_files_no = [file.split("_")[0] for file in os.listdir(path_to_out_files)]

    for file, path in files_dict.items():
        print(file)
        file_sno = file.split("_")[0]
        if file_sno in ran_files_no:
            continue
        site_df_fil_row = site_df_fil.loc[
            lambda df: df.file_nm.str.split("_", expand=True)[0] == file_sno
        ]
        file_full_path = os.path.join(path, file)
        time_threshold = site_df_fil_row.time_thresholds.values[0].split("-")
        file_path, start, time = define_defaults(
            file_=file_full_path, time_threshold_=time_threshold
        )

        add_ramp_volume_to_mainline = False
        if file_sno in ramp_files_dict.keys():
            ramp_file_path = ramp_files_dict[file_sno]
            ramp_volume, ramp_time = read_ramp_data(
                ramp_file_path, time, is_flow_volume=is_flow_volume_dict[file]
            )
            add_ramp_volume_to_mainline = True
        else:
            add_ramp_volume_to_mainline = False
            ramp_volume, ramp_time = [], []  # Add empty values. Not using it for now.

        if file == "99_Lane Drop Diverge_2.csv":
            remove_lane_data = [2, 5]
        else:
            remove_lane_data = []

        speed_1, volume_1, date_time, lanes = read_file_data(
            file_path,
            time,
            is_flow_volume=is_flow_volume_dict[file],
            remove_lane_data_=remove_lane_data,
        )
        volume_1, speed_1, ramp_volume_1, approach_time = approach_sped_volume_ramp(
            speed_1,
            volume_1,
            date_time,
            lanes,
            ramp_volume,
            ramp_time,
            add_ramp_volume_to_mainline,
        )

        # Step 1
        if len(site_df_fil_row) == 1:
            if int(site_df_fil_row.fix_ffs) == 1:
                FFS = site_df_fil_row.ffs.values[0]
            else:
                FFS = FFS_fn(volume_1, speed_1, 4)
        else:
            raise ValueError("Check site summary data")

        # Step 2
        (
            breakdown_index,
            breakdown_volume,
            breakdown_speed,
            uncong_index,
            uncong_speed,
            uncong_volume,
            flow_seq,
            seq,
            volume,
            speed,
            ramp_uncong,
            ramp_bd,
        ) = breakdown_fn(speed_1, volume_1, ramp_volume_1, FFS, 4)

        # Step 3
        pre_breakdown_index, post_breakdown_index = pre_post_breakdown_index_fn(
            breakdown_index, 0
        )
        # step 4
        file_comp = file.split(".")
        file_comp[0] = file_comp[0] + "_pre_brkdn"
        out_file = ".".join(file_comp)
        out_file_path = os.path.join(path_to_out_files, out_file)
        print_pre_BD_data(
            speed,
            volume,
            ramp_volume_1,
            approach_time,
            pre_breakdown_index,
            out_file_path,
        )

    # show_figures(start)
