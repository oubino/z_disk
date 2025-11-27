import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Filter localisation data based on localisation precision"
    )

    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        type=str,
        help="name of the experiment",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--localisation_precision_filter",
        action="store",
        type=float,
        default=5.0,
        help="filter out localisations with precision larger (i.e. worse) than this value",
        required=True,
    )

    args = parser.parse_args(argv)

    folder = os.path.join("experiments", args.experiment, "output")

    input_folder = os.path.join(folder, "segmented_z_disks_denoised")
    output_folder = os.path.join(folder, "segmented_z_disks_denoised_filtered")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    # calculate localisation precision before filtering

    x_prec = []
    y_prec = []
    z_prec = []

    for file in files:

        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

        x_prec.append(df["Group Sigma X Pos"])
        y_prec.append(df["Group Sigma Y Pos"])
        z_prec.append(df["Group Sigma Z"])

    x_prec = np.concatenate(x_prec)
    y_prec = np.concatenate(y_prec)
    z_prec = np.concatenate(z_prec)

    x_prec = [float(x.lstrip(" ")) for x in x_prec]
    y_prec = [float(y.lstrip(" ")) for y in y_prec]
    z_prec = [float(z.lstrip(" ")) for z in z_prec]

    pd_df = pd.DataFrame(
        {
            "X (nm)": x_prec,
            "Y (nm)": y_prec,
            "Z (nm)": z_prec,
        }
    )

    # Use Arial font and set global font sizes
    plt.rcParams.update(
        {
            "axes.titlesize": 16,  # title size
            "axes.labelsize": 14,  # x/y label size
            "xtick.labelsize": 12,  # x tick label size
            "ytick.labelsize": 12,  # y tick label size
            "legend.fontsize": 12,  # legend label size
        }
    )

    plt.figure(figsize=(10, 6))
    plot = sns.histplot(
        data=pd_df,
        element="step",
        stat="count",
    )

    legend = plot.get_legend()
    legend.set_title(None)
    plt.xlim(0, 50)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.xlabel("Localisation precision")
    plt.ylabel("Localisation count")
    hist_loc = os.path.join(output_folder, "loc_prec_histogram.svg")
    plt.savefig(hist_loc, bbox_inches="tight", transparent=True)

    print("Mean precision (pre-filtering)")
    print(pd_df.mean())

    print("Median precision (pre-filtering)")
    print(pd_df.median())

    mean_precision_list = []
    dropped_files = []
    input_locs = 0
    output_locs = 0

    # change output folder so can save files
    output_folder = os.path.join(
        output_folder, f"{args.localisation_precision_filter}nm_filter"
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in files:

        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

        input_locs += len(df)

        df = df.filter(
            pl.col("Group Sigma X Pos").str.strip_chars_start(" ").cast(pl.Float64)
            < args.localisation_precision_filter
        )
        df = df.filter(
            pl.col("Group Sigma Y Pos").str.strip_chars_start(" ").cast(pl.Float64)
            < args.localisation_precision_filter
        )
        df = df.filter(
            pl.col("Group Sigma Z").str.strip_chars_start(" ").cast(pl.Float64)
            < args.localisation_precision_filter
        )

        mean_x_precision = (
            df.select(
                pl.col("Group Sigma X Pos").str.strip_chars_start(" ").cast(pl.Float64)
            )
            .mean()
            .item()
        )
        mean_y_precision = (
            df.select(
                pl.col("Group Sigma Y Pos").str.strip_chars_start(" ").cast(pl.Float64)
            )
            .mean()
            .item()
        )
        mean_z_precision = (
            df.select(
                pl.col("Group Sigma Z").str.strip_chars_start(" ").cast(pl.Float64)
            )
            .mean()
            .item()
        )

        if len(df) > 0:
            mean_precision_list.append(
                (mean_x_precision + mean_y_precision + mean_z_precision) / 3
            )
            output_locs += len(df)
            df.write_csv(os.path.join(output_folder, file))
        else:
            dropped_files.append(file)

    print("Mean precision (after filtering): ", np.mean(mean_precision_list))

    print("Dropped files: ", dropped_files)
    print("Number of dropped files: ", len(dropped_files))
    print("Number of retained files: ", len(files) - len(dropped_files))
    print("% of localisations kept: ", 100 * output_locs / input_locs)


if __name__ == "__main__":
    main()
