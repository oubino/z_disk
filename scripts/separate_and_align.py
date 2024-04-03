"""Separate and align module

Module takes in the point cloud and separates into each z-disk and optionally can align
"""

import argparse
import os
import polars as pl
import numpy as np
from sklearn.decomposition import PCA

def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If try to convert but already files there"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Separate and align z disks"
    )

    parser.add_argument(
        "-a",
        "--align",
        action="store_true",
        help="whether to align data",
    )

    args = parser.parse_args(argv)

    input_folder = "output/segmented_pointclouds"
    output_folder = "output/segmented_z_disks"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    for file in files:
        
        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

        z_disks = df.partition_by("gt_label")

        for index, z_disk in enumerate(z_disks):

            aligned = False

            if args.align:

                # D x N shape i.e. 3 x Number of points
                array = np.array((z_disk["x"], z_disk["y"], z_disk["z"]))
                # N x D
                array = np.swapaxes(array, 0, 1)

                # if less than 3 samples
                if len(array) < 3:
                    print('Not enough samples therefore skipping this z disk')
                
                else:
                    pca = PCA(n_components=3)
                    # N x D
                    array = pca.fit_transform(array)
                    # D x N
                    array = np.swapaxes(array, 0, 1)

                    z_disk = z_disk.with_columns(pl.Series(name='x', values=array[0,:]))
                    z_disk = z_disk.with_columns(pl.Series(name='y', values=array[1,:]))
                    z_disk = z_disk.with_columns(pl.Series(name='z', values=array[2,:]))
                    aligned = True

            # save 
            if not aligned:
                save_path = os.path.join(output_folder, f"{file.rstrip('.csv')}_zdisk_{index}.csv")
            else:
                save_path = os.path.join(output_folder, f"{file.rstrip('.csv')}_zdisk_{index}_aligned.csv")
            z_disk.write_csv(save_path)

if __name__ == "__main__":
    main()