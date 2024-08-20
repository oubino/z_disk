"""Separate and align module

Module takes in the point cloud and separates into each z-disk and optionally can align
"""

import argparse
import os
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as Rot
import warnings

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
    files = [f for f in files if f.endswith('.csv')]

    for file in files:
        
        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

        z_disks = df.partition_by("gt_label")

        if args.align:
            warnings.warn("Note that distances between localisations will be changed on order of 10^-11 units i.e. very small error")

        for index, z_disk in enumerate(z_disks):

            aligned = False

            if args.align:

                # D x N shape i.e. 2 x Number of points
                array_full = np.array((z_disk["x"], z_disk["y"], z_disk["z"]))
                array = np.array((z_disk["x"], z_disk["y"]))
                # N x D
                array = np.swapaxes(array, 0, 1)
                array_full = np.swapaxes(array_full, 0, 1)

                # check
                a = array_full[:,2]

                # if less than 2 samples
                if len(array) < 2:
                    print('Not enough samples therefore skipping this z disk')
                
                else:
                    pca = PCA(n_components=2)
                    # N x D
                    pca = pca.fit(array)
                    # calculate rotation matrix between maximal PCA compoment and x axis
                    pca_vector = np.append(pca.components_[0], 0.0)
                    rot, _ = Rot.align_vectors(np.array([1,0,0]), pca_vector)
                    # rotate points
                    array_full = rot.apply(array_full)
                    # D x N
                    array_full = np.swapaxes(array_full, 0, 1)

                    z_disk = z_disk.with_columns(pl.Series(name='x', values=array_full[0,:]))
                    z_disk = z_disk.with_columns(pl.Series(name='y', values=array_full[1,:]))
                    z_disk = z_disk.with_columns(pl.Series(name='z', values=array_full[2,:]))

                    aligned = True

                    # check
                    b = array_full[2,:]
                    a = np.round(a, 6)
                    b = np.round(b, 6)
                    assert np.array_equal(a,b)

            # save 
            if not aligned:
                save_path = os.path.join(output_folder, f"{file.rstrip('.csv')}_zdisk_{index+1}.csv")
            else:
                save_path = os.path.join(output_folder, f"{file.rstrip('.csv')}_zdisk_{index+1}_aligned.csv")   
            z_disk.write_csv(save_path)

if __name__ == "__main__":
    main()