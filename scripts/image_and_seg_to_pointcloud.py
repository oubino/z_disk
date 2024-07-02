"""Image with segmentation to pointcloud module

Module takes in the .npy segmentation and returns datastructure with segmentation
"""

import os
from base import item as item_class
import numpy as np
#import time


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If try to convert but already files there"""

    folder = "output"
    
    input_datastructure_folder = os.path.join(folder, "datastructures")
    input_segmentation_folder = os.path.join(folder, "segmentations")
    output_folder = os.path.join(folder, "segmented_pointclouds")

    if not os.path.exists(output_folder):
          os.makedirs(output_folder)

    files = os.listdir(input_datastructure_folder)

    for file in files:
            
            item = item_class(None, None, None)
            item.load_from_parquet(os.path.join(input_datastructure_folder, file))

            seg_loc = os.path.join(input_segmentation_folder, item.name + ".npy")
            seg = np.load(seg_loc)

            # ilastik_seg is [z,y,x,c] where channel 0 is segmentation
            # where each integer represents different instance of a cell
            # i.e. 1 = one cell; 2 = different cell; etc.
            seg = seg[:, :, :, 0]

            output_loc = os.path.join(output_folder, item.name + ".csv")

            # save instance mask to dataframe
            df = item.mask_pixel_2_coord(seg)
            item.df = df
            item.save_df_to_csv(
                output_loc,
                drop_zero_label=True,
            )

if __name__ == "__main__":
    main()