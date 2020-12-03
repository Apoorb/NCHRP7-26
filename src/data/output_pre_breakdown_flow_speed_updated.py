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

# Reference
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


# Read data
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
    # print("Filtered Start Time: " + str(time[0].time()))
    # print("Filtered End Time: " + str(time[1].time()))

    return (file_path, start, time)


def read_file_data(file_path, time, is_flow_volume):
    raw_data = pd.read_csv(file_path, header=0)
    if is_flow_volume == "flow":
        raw_data.Volume = raw_data.Volume / 4

    if sum(raw_data.duplicated(["Timestamp", "Lane"], keep=False)) != 0:
        duplicated_data = raw_data[
            raw_data.duplicated(["Timestamp", "Lane"], keep=False)
        ]
        raw_data = raw_data[~raw_data.duplicated(["Timestamp", "Lane"], keep="first")]
    num_lanes = raw_data.Lane.max()
    raw_data_mainline = raw_data.loc[lambda df: df.Lane != -1]  # Just use mainline
    raw_data_mainline.Timestamp = pd.to_datetime(raw_data_mainline.Timestamp)
    # print(raw_data_mainline)
    raw_data_mainline_pivot = raw_data_mainline.pivot(
        index="Timestamp", columns="Lane", values=["Volume", "Speed"]
    )
    # print(raw_data_mainline_pivot)
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


# Analysis
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
            approach_speed.append((nr / dr))
            approach_vol.append((volume / len(volume_1)))

    # print((approach_speed))
    # print((approach_vol))
    # print((approach_time))
    # print(ramp_time)
    ramp_vol = []
    if len(ramp_volume) != 0:
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
    # classifying thespeed flow curve into 4 regions
    # Region 1: > 0.9 FFS, Region 2: > 0.75 * FFS & <= 0.9 * FFS, Region 3: <= 0.75 * FFS, Region 4 (after breakdown): 0.75 * FFS and <= 0.9 * FFS
    # the data classifier variable will have 1,2,3 & 4 at corresponding indexes of volume, speed and time
    data_classifier = []
    recovery = 0
    if len(ramp_volume) != 0:
        for i in range(len(volume)):
            speed[i] = float(speed[i])
            volume[i] = float(volume[i]) * x
            ramp_volume[i] = float(ramp_volume[i]) * x
            if recovery != 1:
                if float(speed[i]) > 0.75 * FFS and float(speed[i]) <= 0.9 * FFS:
                    data_classifier.append(2)
                elif float(speed[i]) <= 0.75 * FFS:
                    data_classifier.append(3)
                    recovery = 1
                else:
                    data_classifier.append(1)
            else:
                if float(speed[i]) > 0.9 * FFS:
                    data_classifier.append(1)
                    recovery = 0
                elif float(speed[i]) > 0.75 * FFS and float(speed[i]) <= 0.9 * FFS:
                    data_classifier.append(4)
                else:
                    data_classifier.append(3)
    else:
        for i in range(len(volume)):
            speed[i] = float(speed[i])
            volume[i] = float(volume[i]) * x
            if recovery != 1:
                if float(speed[i]) > 0.75 * FFS and float(speed[i]) <= 0.9 * FFS:
                    data_classifier.append(2)
                elif float(speed[i]) <= 0.75 * FFS:
                    data_classifier.append(3)
                    recovery = 1
                else:
                    data_classifier.append(1)
            else:
                if float(speed[i]) > 0.9 * FFS:
                    data_classifier.append(1)
                    recovery = 0
                elif float(speed[i]) > 0.75 * FFS and float(speed[i]) <= 0.9 * FFS:
                    data_classifier.append(4)
                else:
                    data_classifier.append(3)

    return (volume, speed, ramp_volume, data_classifier)


def flowrate_sequence(volume, data_classifier):
    # generating the flowrate sequence based on the highest flowrate
    volume_classified = []
    for i in range(len(data_classifier)):
        if data_classifier[i] == 1:
            volume_classified.append(volume[i])

    seq, round_100 = roundup(max(volume_classified))
    flow_seq = np.linspace(1000, round_100, seq - 10)
    # print(flow_seq)
    return flow_seq


def counts_ranges(volume, speed, time, data_classifier, classifier, flow_seq):
    volume_classified = []
    speed_classified = []
    time_classified = []
    for j in range(
        len(classifier)
    ):  # segregating uncongested flow data from region 1 & 2
        for i in range(len(data_classifier)):
            if data_classifier[i] == classifier[j]:
                volume_classified.append(volume[i])
                speed_classified.append(speed[i])
                time_classified.append(time[i])

    counts = []
    for i in range(1, len(flow_seq)):  # binning the values
        temp = 0
        for j in range(len(volume_classified)):
            if (
                volume_classified[j] >= flow_seq[i - 1]
                and volume_classified[j] < flow_seq[i]
            ):
                temp = temp + 1
        counts.append(temp)
    # print(counts)
    return (volume_classified, speed_classified, time_classified, counts)


def counts_ranges_2(volume, flow_seq):
    counts = []
    for i in range(1, len(flow_seq)):  # bining the values
        temp = 0
        for j in range(len(volume)):
            if volume[j] >= flow_seq[i - 1] and volume[j] < flow_seq[i]:
                temp = temp + 1
        counts.append(temp)
    # print(counts)
    return counts


def avg_flowrate_fn(flow_seq):
    avg_flowrate = []
    for i in range(len(flow_seq) - 1):  # Calculating the avergae flowrate
        avg_flowrate.append((flow_seq[i] + flow_seq[i + 1]) / 2)
    # print(avg_flowrate)
    return avg_flowrate


def pre_breakdown_fn(volume, speed, approach_time, data_classifier):
    temp = 2
    pre_bd_classifier = []
    for i in range(
        len(data_classifier)
    ):  # the first index of a region region 3 succeding a reion 1 or region 2. Ex: 1,1,1,2,2,2,2,3 or 1,1,1,1,3 Region 3 succeeding a region 4 will not be a breakdown
        if temp == 2 and (data_classifier[i] == 2 or data_classifier[i] == 1):
            pre_bd_classifier.append(0)
            temp = 3
        elif temp == 3 and data_classifier[i] == 3:
            pre_bd_classifier.append(1)
            temp = 2
        else:
            pre_bd_classifier.append(0)

    pre_bd_volume = []
    pre_bd_speed = []
    pre_bd_time = []
    for j in range(
        1, len(pre_bd_classifier)
    ):  # Since first index of the Region 3 is identified, j-1 gives us the Prebreakdown value
        if pre_bd_classifier[j] == 1:
            pre_bd_volume.append(volume[j - 1])
            pre_bd_speed.append(speed[j - 1])
            pre_bd_time.append(approach_time[j - 1])

    # print(pre_bd_volume)
    # print(pre_bd_speed)
    # print(pre_bd_time)

    return (pre_bd_volume, pre_bd_speed, pre_bd_time, pre_bd_classifier)


def breakdown_prob(pre_breakdown_counts, uncong_counts, flow_seq):
    # calculating the breakdown probability
    breakdown_probability = []
    breakdow_flow_seq = []
    for i in range(len(pre_breakdown_counts)):
        if uncong_counts[i] != 0:
            breakdown_probability.append(
                round(float(pre_breakdown_counts[i] / float(uncong_counts[i])) * 100, 1)
            )
            breakdow_flow_seq.append(flow_seq[i])
        # else:
        #     breakdown_probability.append(0)

    # print(breakdown_probability, breakdow_flow_seq)
    return (breakdown_probability, breakdow_flow_seq)


def weibull_fn(flow_seq, breakdown_probability):
    removals = np.where(
        np.array(breakdown_probability) == 0
    )  # identifying breakdowns with 0% prob.
    removals = removals[0]
    if len(removals) > 0:
        inc = 0
        for j in range(
            len(removals)
        ):  # removing breakdown probs with 0% and their corresponding flowrate value
            del breakdown_probability[removals[j] - inc]
            del flow_seq[removals[j] - inc]
            inc += 1
    for i in range(len(breakdown_probability)):  # converting prob with 100 to 99.9
        if breakdown_probability[i] > 0 and breakdown_probability[i] <= 100:
            if breakdown_probability[i] == 100:
                breakdown_probability[i] = 99.9
    X = []
    Y = []
    for i in range(len(breakdown_probability)):
        X.append(np.log(flow_seq[i]))
        Y.append(np.log(-np.log(1 - breakdown_probability[i] / 100)))
    slope, intercept, r_val, p_val, std_err = linregress(X, Y)
    beta = slope
    alpha = np.exp(intercept / (-beta))
    # print(str(alpha) + " " + str(beta))
    weibull_x = np.linspace(100, 3000, 2000)
    fit_y = intercept + slope * np.log(weibull_x)
    weibull_vals = 1 - np.exp(-np.exp(fit_y))
    # print(weibull_vals, weibull_x, alpha, beta)
    return (weibull_vals, weibull_x, alpha, beta)


def estimated_capacity_fn(alpha, beta):
    estimated_capacity = alpha * 0.163 ** (1 / beta)
    return round(estimated_capacity)


# Output
def output_file(
    database,
    date_time,
    FFS,
    volume,
    pre_breakdown_counts,
    estimated_capacity,
    alpha,
    beta,
):
    filename = "scratch/Data_output.csv"
    if os.path.isfile(filename):
        with open("scratch/Data_output.csv", "a") as f:
            f.write(
                database.split("_")[0]
                + ","
                + database.split(".csv")[0]
                + ","
                + str(date_time[0])
                + ","
                + str(date_time[len(date_time) - 1])
                + ","
                + str(FFS)
                + ","
                + str(len(volume))
                + ","
                + str(sum(pre_breakdown_counts))
                + ","
                + str(alpha)
                + ","
                + str(beta)
                + ","
                + str(estimated_capacity)
                + ","
                + "\n"
            )
    else:
        with open("scratch/Data_output.csv", "a") as f:
            f.write(
                "Sl No.,File,Start Date,End Date,FFS,Total Counts,Breakdown Events,Alpha,Beta,Estimated Capacity\n"
            )
            f.close()
        output_file(
            database,
            date_time,
            FFS,
            volume,
            pre_breakdown_counts,
            estimated_capacity,
            alpha,
            beta,
        )


def print_pre_BD_data(pre_bd_volume, pre_bd_speed, pre_bd_time, file):
    file_comp = file.split(".")
    file_comp[0] = file_comp[0] + "_pre_brkdn"
    out_file = ".".join(file_comp)
    out_file_path = os.path.join(path_to_out_files, out_file)
    df_content = {
        "Time": pre_bd_time,
        "MainlineSpeed": pre_bd_speed,
        "MainlineVol": pre_bd_volume,
    }
    df = pd.DataFrame(df_content)
    df.to_csv(out_file_path, index=False)


def print_uncongested_data(uncong_volume, uncong_speed, uncong_time, file):
    file_comp = file.split(".")
    file_comp[0] = file_comp[0] + "_uncongested"
    out_file = ".".join(file_comp)
    out_file_path = os.path.join(path_to_out_files, out_file)
    df_content = {
        "Time": uncong_time,
        "MainlineSpeed": uncong_speed,
        "MainlineVol": uncong_volume,
    }
    df = pd.DataFrame(df_content)
    df.to_csv(out_file_path, index=False)


# Plotting
def cumm_breakdown_prob(counts):
    cumm_pre_breakdown_distr = []
    cumm_pre_breakdown = []
    cumm_pre_breakdown.append(0)
    for i in range(len(counts)):
        cumm_pre_breakdown.append((cumm_pre_breakdown[i] + counts[i]))
    for i in range(len(cumm_pre_breakdown)):
        cumm_pre_breakdown_distr.append(
            round(cumm_pre_breakdown[i] / float(np.sum(counts)) * 100, 1)
        )
    return (cumm_pre_breakdown, cumm_pre_breakdown_distr)


def plot_figure_1(volume, speed, pre_breakdown_speed, pre_breakdown_flowrate, file):
    global annot, fig, ax, sc
    file_comp = file.split(".")[0]
    out_file_path = os.path.join(path_to_out_files, file_comp)
    plt.figure()
    plt.scatter(
        [volume],
        [speed],
        label="Speed-Flow Curve",
        color="cornflowerblue",
        marker=".",
        s=12,
    )
    sc = plt.scatter(
        [pre_breakdown_flowrate],
        [pre_breakdown_speed],
        label="Prebreakdown Events",
        color="red",
        marker="x",
        s=12,
    )
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("Flow Rate (veh/hr/ln)", fontsize=12)
    plt.ylabel("Speed (mph)", fontsize=12)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(5.5, 3)
    plt.savefig(out_file_path + "_time_1.png", bbox_inches="tight")


def plot_figure_2(flow_seq, cumm_pre_breakdown_distr, weibull_vals, weibull_x, file):
    file_comp = file.split(".")[0]
    out_file_path = os.path.join(path_to_out_files, file_comp)
    plt.figure()
    plt.plot(
        flow_seq,
        cumm_pre_breakdown_distr,
        label="Observed Prebreakdown CDF",
        color="red",
        linestyle="--",
    )
    plt.plot(
        weibull_x, weibull_vals, label="Fitted Weibull CDF", color="cornflowerblue"
    )
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("Flow Rate (veh/hr/ln)", fontsize=12)
    plt.ylabel("Probability of Breakdown", fontsize=12)
    plt.ylim([0, 1.1])
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(5.5, 3)
    plt.savefig(out_file_path + "_time_2.png", bbox_inches="tight")


def show_figures(start):
    print("Code run time: " + str(datetime.datetime.utcnow() - start))
    # plt.show()


if __name__ == "__main__":
    path_to_site_summary = (
        r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - NCHRP 07-26"
        r"\Task 6 - Execute Phase II Plan\Site Analysis & Metadata"
        r"\NCHRP07-26_Site_summary_shared.xlsx"
    )
    path_to_kai_clean_data = (
        r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - NCHRP 07-26"
        r"\Task 6 - Execute Phase II Plan\site_summary_template\05_organized_KAI_Data"
    )
    path_to_uw_clean_data = (
        r"C:\Users\abibeka\Kittelson & Associates, Inc\Burak Cesme - NCHRP 07-26"
        r"\Task 6 - Execute Phase II Plan\site_summary_template\04_organized_UW_Data"
    )
    path_to_weave_ramp_data = os.path.join(path_to_kai_clean_data, "weave_ramp_volumes")
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
    # print("Do you want to re-run previously ran file: (y/n)")
    # user_input = input()
    user_input = "y"

    if user_input == "y":
        ran_files_no = []
    else:
        ran_files_no = [file.split("_")[0] for file in os.listdir(path_to_out_files)]

    i = 0
    for file, path in files_dict.items():
        if file == "99_Lane Drop Diverge_2.csv":
            continue
        i += 1
        print(i)
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

        # Reading File data
        speed_1, volume_1, date_time, lanes = read_file_data(
            file_path, time, is_flow_volume=is_flow_volume_dict[file]
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

        # Identifying FFS from site summary sheet
        if len(site_df_fil_row) == 1:
            if int(site_df_fil_row.fix_ffs) == 1:
                FFS = site_df_fil_row.ffs.values[0]
            else:
                FFS = FFS_fn(volume_1, speed_1, 4)
        else:
            raise ValueError("Check site summary data")

        # FFS=75.3 # this is uncommented and varied while testing different FFS to fix the value if the site summary sheet value is not good

        volume, speed, ramp_volume, data_classifier = breakdown_fn(
            speed_1, volume_1, ramp_volume_1, FFS, 4
        )
        flow_seq = flowrate_sequence(volume, data_classifier)
        uncong_volume, uncong_speed, uncong_time, uncong_counts = counts_ranges(
            volume, speed, approach_time, data_classifier, [1, 2], flow_seq
        )
        avg_flowrate = avg_flowrate_fn(flow_seq)
        pre_bd_volume, pre_bd_speed, pre_bd_time, pre_bd_classifier = pre_breakdown_fn(
            volume, speed, approach_time, data_classifier
        )
        pre_bd_counts = counts_ranges_2(pre_bd_volume, flow_seq)
        breakdown_probability, breakdown_flow_seq = breakdown_prob(
            pre_bd_counts, uncong_counts, flow_seq
        )
        weibull_vals, weibull_x, alpha, beta = weibull_fn(
            avg_flowrate, breakdown_probability
        )
        print("Alpha: " + str(round(alpha)) + " Beta: " + str(round(beta, 1)))
        estimated_capacity = estimated_capacity_fn(alpha, beta)
        print("Estimated Capacity: " + str(estimated_capacity))

        # Plotting
        cumm_pre_breakdown, cumm_pre_breakdown_distr = cumm_breakdown_prob(
            pre_bd_counts
        )
        plot_figure_1(volume, speed, pre_bd_speed, pre_bd_volume, file)
        plot_figure_2(
            flow_seq,
            np.divide(cumm_pre_breakdown_distr, 100),
            weibull_vals,
            weibull_x,
            file,
        )
        print()
        output_file(
            file,
            date_time,
            round(FFS, 2),
            volume,
            pre_bd_counts,
            estimated_capacity,
            round(alpha, 0),
            round(beta, 1),
        )
        print_pre_BD_data(pre_bd_volume, pre_bd_speed, pre_bd_time, file)
        print_uncongested_data(uncong_volume, uncong_speed, uncong_time, file)
        if i == 73:  # First 73 are KAI sites
            show_figures(start)
            break
