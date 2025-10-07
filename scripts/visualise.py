"""Script for visualising the data"""

# Take in datastructure and visualise as points or as images
import argparse
from collections import Counter
import os

import matplotlib.colors as cl
import numpy as np
import open3d as o3d
import polars as pl

def load_file(file, x_name, y_name, z_name, channel_name):
    """Load file for visualisation

    Args:
        file (string): Name of the file to read in
        x_name (string): Name of the x column
        y_name (string): Name of the y column
        z_name (string): Name of the z column
        channel_name (string): Name of the channel column

    Returns:
        df (pl.DataFrame): DataFrame that has been loaded in
        df[channel_name].unique() (list): List of unique channels in the
            dataframe
    """

    if file.endswith('.parquet'):
        df = pl.read_parquet(file, columns=[x_name, y_name, z_name, channel_name])
    elif file.endswith('.csv'):
        df = pl.read_csv(file, columns=[x_name, y_name, z_name, channel_name])
    else:
        raise ValueError("File must be parquet or csv file")

    return df, df[channel_name].unique()


def add_pcd(
    df,
    chan,
    x_name,
    y_name,
    z_name,
    chan_name,
    unique_chans,
    cmap,
    pcds,
    spheres,
    sphere_size,
):
    """Add a parquet file as a PCD for visualisation

    Args:
        df (pl.DataFrame): DataFrame containing data to visualise
        chan (int): Channel to visualise
        x_name (str): Name of the x column in the dataframe
        y_name (str): Name of the y column in the dataframe
        z_name (str): Name of the z column in the dataframe
        chan_name (str): Name of the protein imaged in the channel
        unique_chans (list): Unique channels in the dataframe
        cmap (string): Colour to visualise the channel in
        pcds (list): List of pcds that will be visualised
        spheres (bool) : Whether to plot points as spheres
        sphere_size (float) : Size of spheres to plot

    Returns:
        pcds (list): List of pcds that has been updated and will be visualised
    """
    if chan in unique_chans:
        pcd = o3d.geometry.PointCloud()
        coords = (
            df.filter(pl.col(chan_name) == chan)
            .select([x_name, y_name, z_name])
            .to_numpy()
        )

        if not spheres:
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.paint_uniform_color(cl.to_rgb(cmap[chan]))
        else:
            spheres = []
            for point in np.asarray(coords):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
                sphere.translate(point)
                sphere.paint_uniform_color(cl.to_rgb(cmap[chan]))
                spheres.append(sphere)

            # Combine all spheres into one mesh
            pcd = spheres[0]
            for sphere in spheres[1:]:
                pcd += sphere
        pcds.append(pcd)
    return pcds


class Present:
    """Required for visualising the parquet file"""

    def __init__(self):
        self.chan_present = [True, True, True, True]


def visualise_file(
    file_loc,
    x_name,
    y_name,
    z_name,
    channel_name,
    channel_labels,
    cmap=["r", "darkorange", "b", "y"],
    spheres=False,
    sphere_size=0.1,
    dbscan=False,
    dbscan_params={},
):
    """Visualise file

    Args:
        file_loc (str) : Location of the file to visualise
        x_name (str) : Name of the x column in the data
        y_name (str) : Name of the y column in the data
        z_name (str) : Name of the z column in the data, if is None, then
            assumes data is 2D
        channel_name (str) : Name of the channel column in the data
        channel_labels (dict) : Dictionary mapping channel label to name
        cmap (list) : CMAP to visualise the data
        spheres (bool) : Whether to visualise as spheres
        sphere_size (float) : Size of the spheres"""

    df, unique_chans = load_file(file_loc, x_name, y_name, z_name, channel_name)

    pcds = []

    cmap = ["r", "darkorange", "b", "y"]
    for key in channel_labels.keys():
        pcds = add_pcd(
            df,
            key,
            x_name,
            y_name,
            z_name,
            channel_name,
            unique_chans,
            cmap,
            pcds,
            spheres,
            sphere_size,
        )

    labels = visualise(pcds, None, None, None, unique_chans, channel_labels, cmap, dbscan, dbscan_params)

    # check each point in dataframe is labelled
    assert len(df) == len(labels)
    
    return df, labels

def visualise(
    pcds,
    locs_to_locs,
    locs_to_clusters,
    clusters_to_clusters,
    unique_chans,
    channel_labels,
    cmap,
    dbscan,
    dbscan_params={},
):
    """Visualise point cloud data

    Args:
        pcds (list) : List of point cloud data files
        locs_to_locs (list) : Lines to draw between localisations
        locs_to_clusters (list) : Lines to draw from localisations to clusters
        clusters_to_clusters (list) : Lines to draw between clusters
        unique_chans (list) : List of unique channels
        channel_labels (dict) : Dictionary mapping channel index to real name
        cmap (list) : Colours to plot in"""

    _ = o3d.visualization.Visualizer()

    assert len(pcds) == len(unique_chans)

    # pcd = df_to_feats(pcd, csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)', 1000)

    present = Present()

    def visualise_chan_0(vis):
        """Function needed for key binding to visualise channel 0

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.chan_present[0]:
            vis.remove_geometry(pcds[0], False)
            present.chan_present[0] = False
        else:
            vis.add_geometry(pcds[0], False)
            present.chan_present[0] = True

    def visualise_chan_1(vis):
        """Function needed for key binding to visualise channel 1

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.chan_present[1]:
            vis.remove_geometry(pcds[1], False)
            present.chan_present[1] = False
        else:
            vis.add_geometry(pcds[1], False)
            present.chan_present[1] = True

    def visualise_chan_2(vis):
        """Function needed for key binding to visualise channel 2

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.chan_present[2]:
            vis.remove_geometry(pcds[2], False)
            present.chan_present[2] = False
        else:
            vis.add_geometry(pcds[2], False)
            present.chan_present[2] = True

    def visualise_chan_3(vis):
        """Function needed for key binding to visualise channel 3

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.chan_present[3]:
            vis.remove_geometry(pcds[3], False)
            present.chan_present[3] = False
        else:
            vis.add_geometry(pcds[3], False)
            present.chan_present[3] = True

    # reverse pcds for visualisation
    # pcds.reverse()

    key_to_callback = {}
    key_to_callback[ord("K")] = visualise_chan_0
    key_to_callback[ord("R")] = visualise_chan_1
    key_to_callback[ord("T")] = visualise_chan_2
    key_to_callback[ord("Y")] = visualise_chan_3

    if 0 in unique_chans:
        print(f"Channel 0 is {channel_labels[0]} is colour {cmap[0]} to remove use K")
    if 1 in unique_chans:
        print(f"Channel 1 is {channel_labels[1]} is colour {cmap[1]} to remove use R")
    if 2 in unique_chans:
        print(f"Channel 2 is ... is colour {cmap[2]} to remove use T")
    if 3 in unique_chans:
        print(f"Channel 3 is ... is colour {cmap[3]} to remove use Y")

    if locs_to_clusters is not None:
        pcds.append(locs_to_clusters)
    if clusters_to_clusters is not None:
        pcds.append(clusters_to_clusters)
    if locs_to_locs is not None:
        pcds.append(locs_to_locs)
        
    # dbscan
    if dbscan:
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcds[0].cluster_dbscan(eps=dbscan_params["eps"], min_points=dbscan_params["min_pts"], print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        assert len(pcds) == 1 # only works for one channel

        remove_outliers = input("Use DBSCAN to remove outliers? (Y) ")
        if remove_outliers == "Y":
            counter = Counter(labels)
            assert set(counter.keys()) == {0,-1} # check only 1 cluster
            assert counter[0]/(counter[0] + counter[-1]) > 0.95 # check >95% points in cluster
            print(f"Visualising with {counter[-1]} outlier point(s) in RED")
            colors = np.array([[0,0,0,1] for _ in labels])
            colors[labels < 0] = [1,0,0,1]
        else:
            import matplotlib.pyplot as plt
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
        pcds[0].colors = o3d.utility.Vector3dVector(colors[:, :3].copy())

    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)

    if remove_outliers == "Y":
        remove_outliers = input("Are you happy with the points in red (outliers) being removed? (Y) ")
        if remove_outliers == "Y":
            return labels
        else:
            return None
    else:
        return None

def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If incorrect file type for input file"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Visualise the data from a .csv or .parquet file"
    )

    parser.add_argument(
        "-i",
        "--input_file",
        action="store",
        type=str,
        help="location of the input file (either .parquet or .csv)",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dbscan",
        nargs=2,
        type=float,
        help="if specified, takes two numbers: eps and min_samples",
        default=None,
    )

    args = parser.parse_args(argv)

    if args.dbscan is None:
        visualise_file(
            args.input_file,
            "x",
            "y",
            "z",
            "channel",
            {0: "channel_0", 1: "channel_1", 2: "channel_2", 3: "channel_3"},
            dbscan=False,
        )
    else:
        dbscan_params = {"eps": args.dbscan[0], "min_pts": int(args.dbscan[1])}
        df, labels = visualise_file(
            args.input_file,
            "x",
            "y",
            "z",
            "channel",
            {0: "channel_0", 1: "channel_1", 2: "channel_2", 3: "channel_3"},
            dbscan=True,
            dbscan_params=dbscan_params,
        )
        if labels is not None: # then we are going to remove the noise points...
            df = df.with_columns(pl.Series(name="noise", values=labels))
            df_without_noise = df.filter(pl.col("noise") != -1).drop("noise")
            if args.input_file.endswith('.parquet'):
                og_df = pl.read_parquet(args.input_file)
            elif args.input_file.endswith('.csv'):
                og_df = pl.read_csv(args.input_file)

            output_df = og_df.join(df_without_noise, how="semi", on=["x", "y", "z", "channel"])

            # check output df
            assert len(output_df) == len(df_without_noise)
            assert output_df.columns == og_df.columns

            # export cleaned df
            filename, file_extension = os.path.splitext(args.input_file)
            output_folder = os.path.join("/".join(filename.split("/")[:-2]), "segmented_z_disks_denoised")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_file = filename.replace("/segmented_z_disks/", "/segmented_z_disks_denoised/") + "_denoised" + file_extension
            if file_extension == ".csv":
                output_df.write_csv(output_file)
            elif file_extension == ".parquet":
                output_df.write_parquet(output_file)

if __name__ == "__main__":
    main()
