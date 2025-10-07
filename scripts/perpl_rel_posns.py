import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from perpl.io import plotting
from perpl.relative_positions import main as calculate_relative_positions
from perpl.relative_positions import getdistances, get_vectors, save_relative_positions


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Apply segmentations back to pointclouds"
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
        "-f",
        "--filter_distance",
        action="store",
        type=int,
        default=150,
        help="filter distance",
    )

    parser.add_argument(
        "-p",
        "--localisation_precision_filter",
        action="store",
        type=float,
        default=5.0,
        help="filter out localisations with precision larger (i.e. worse) than this value",
    )

    parser.add_argument(
        "-nn",
        "--nearest_neighbours",
        action="store",
        type=int,
        default=0,
        help="Number of nearest neighbours to find within the filter distance, "
        "if desired. 0 (default) means no limit on the number of "
        "neighbours used within the filter distance.",
    )

    parser.add_argument(
        "-b",
        "--bin_size",
        action="store",
        type=int,
        default=1,
        help="Size of the bins for plotting the relative positions in histogram",
    )

    parser.add_argument(
        "-v",
        "--visualise",
        action="store_true",
        default=False,
        help="Whether to visualise plots etc.",
    )

    args = parser.parse_args(argv)

    folder = os.path.join("experiments", args.experiment, "output")

    input_folder = os.path.join(folder, "segmented_z_disks_denoised")
    output_folder = os.path.join(folder, "perpl_relative_posns")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    d_values_list = []

    if args.visualise:

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

        plt.hist(x_prec, bins=20, histtype="step", label="x prec")
        plt.hist(y_prec, bins=20, histtype="step", label="y prec")
        plt.hist(z_prec, bins=20, histtype="step", label="z prec")
        plt.legend()
        plt.show()

    mean_precision_list = []

    for file in files:

        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

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
        mean_precision_list.append(
            (mean_x_precision + mean_y_precision + mean_z_precision) / 3
        )

        # Based on lines 845-910 in perpl.relative_positions

        xyz_values = df[["x", "y", "z"]].to_numpy()

        if args.visualise:

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(
                xyz_values[:, 0], xyz_values[:, 1], xyz_values[:, 2], s=1, marker="x"
            )
            ax.set_xlabel("X [nm]")
            ax.set_ylabel("Y [nm]")
            ax.set_zlabel("Z [nm]")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()

        d_values = getdistances(
            xyz_values, args.filter_distance, args.nearest_neighbours, verbose=False
        )[1]

        # Get distances in 2D and 3D for relative positions.
        # Note, get_vectors() is an unhelpful name as it takes the vectors we already have
        # and calculates distances.
        if len(d_values) > 0:
            d_values = get_vectors(d_values, dims=3)
        else:
            raise ValueError("No distances within filter")

        d_values_list.append(d_values)

    d_values = np.concatenate(d_values_list, axis=0)

    info = {
        "results_dir": os.path.join(folder, "perpl_relative_posns"),
        "short_names": False,
        "in_file_no_extension": f"all_z_disks_{args.localisation_precision_filter}precisionfilter",
    }

    ## Plot vector component results
    plotting.plot_histograms(
        d_values,
        dims=3,
        filter_distance=args.filter_distance,
        info=info,
        binsize=args.bin_size,
    )

    ## Save relative positions and vector components.
    xyz_filename = save_relative_positions(
        d_values, args.filter_distance, dims=3, info=info, nns=args.nearest_neighbours
    )

    file_name = xyz_filename.replace("relpos", "locprec").rstrip(".csv") + ".txt"

    np.savetxt(file_name, np.atleast_1d(np.mean(mean_precision_list)))


if __name__ == "__main__":
    main()
