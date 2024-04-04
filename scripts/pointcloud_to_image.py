"""Pointcloud to image module

Module takes in the .txt files and saves as images with associated datastructure
"""

import argparse
import os
import polars as pl
from base import item as item_class
import numpy as np

def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If try to convert but already files there"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Convert pointclouds to images"
    )

    parser.add_argument(
        "-x",
        "--x_col_name",
        action="store",
        type=str,
        help="name of the x column",
        required=True,
    )

    parser.add_argument(
        "-y",
        "--y_col_name",
        action="store",
        type=str,
        help="name of the y column",
        required=True,
    )

    parser.add_argument(
        "-z",
        "--z_col_name",
        action="store",
        type=str,
        help="name of the z column",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--channel_col_name",
        action="store",
        type=str,
        help="name of the channel column (Optional)",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-bs",
        "--bins_or_size",
        action="store",
        type=str,
        help="how to interpret the histo size given as size of bins or number of bins (bins or size)",
        required=True,
        choices=["bins","size"]
    )

    parser.add_argument(
        "-hx",
        "--histo_x_size",
        action="store",
        type=int,
        help="size of histo in x direction",
        required=True,
    )

    parser.add_argument(
        "-hy",
        "--histo_y_size",
        action="store",
        type=int,
        help="size of histo in y direction",
        required=True,
    )

    parser.add_argument(
        "-hz",
        "--histo_z_size",
        action="store",
        type=int,
        help="size of histo in z direction",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--separator",
        action="store",
        type=str,
        help="delimeter that separates the data items in input file (tab or comma)",
        choices = ["tab", "comma"]
        required=True,
    )
    

    args = parser.parse_args(argv)

    input_folder = "data"
    output_folder = "output"

    output_image_folder = os.path.join(output_folder, "images")
    output_datastructure_folder = os.path.join(output_folder, "datastructures")

    histo_size = [args.histo_x_size, args.histo_y_size, args.histo_z_size]

    # create preprocessed directory
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    
    # create preprocessed directory
    if not os.path.exists(output_datastructure_folder):
        os.makedirs(output_datastructure_folder)

    # check with user
    print("List of files which will be converted")
    input_files = os.listdir(input_folder)
    input_files = [os.path.join(input_folder, file) for file in input_files]
    # check file not already present
    for file in input_files:
        file_name = os.path.basename(file)
        if not file_name.endswith(".txt"):
            raise ValueError("Wrong input file name")
        file_name = file_name.rstrip('.txt')
        output_path = os.path.join(output_datastructure_folder, f"{file_name}.parquet")
        if os.path.exists(output_path):
            raise ValueError("Can't preprocess as output file already exists")
    print(input_files)

    # go through files -> convert to datastructure -> save
    for index, file in enumerate(input_files):
        if args.separator == 'comma':
            df = pl.read_csv(file, separator=",")
        elif args.separator == 'tab':
            df = pl.read_csv(file, separator="\t")

        # Get name of file - assumes last part of input file name
        name = os.path.basename(os.path.normpath(file)).removesuffix(".txt")

        # rename x,y,z,channel columns
        df = df.rename(
            {args.x_col_name: "x",
             args.y_col_name: "y",
             args.z_col_name: "z"}
        )

        if df.schema['x'].is_(pl.String):
            df = df.with_columns(pl.col("x").str.strip_chars_start().cast(pl.Float64))

        if df.schema['y'].is_(pl.String):
            df = df.with_columns(pl.col("y").str.strip_chars_start().cast(pl.Float64))

        if df.schema['z'].is_(pl.String):
            df = df.with_columns(pl.col("z").str.strip_chars_start().cast(pl.Float64))

        # if channel column given
        if args.channel_col_name is not None:
            df = df.rename(
                {args.channel_col_name: "channel"}
            )
            if df.schema['channel'].is_(pl.String):
                df = df.with_columns(pl.col("channel").str.strip_chars_start().cast(pl.Int32))
            # Get list of channels - currently takes in all channels
            channel_choice = sorted(list(set(df["channel"])))
        else:
            # add on fake channel column
            df = df.with_columns(
                pl.lit(0).alias("channel")
            )
            channel_choice = [0]

        item =  item_class(
            name,
            df,
            channel_choice,
        )


        item.coord_2_histo(
            histo_size,
            args.bins_or_size,
        )

        # render and save histo
        histo = item.render_histo()

        # ilastik needs channel last and need to tranpose histogram
        # for image space
        # i.e. c,x,y,z
        #      0,1,2,3
        #   -> z,y,x,c
        #      3,2,1,0
        img = np.transpose(histo, (3, 2, 1, 0))

        # all images are saved in yxc
        save_loc = os.path.join(output_image_folder, name + ".npy")
        np.save(save_loc, img)

        item.save_to_parquet(
            output_datastructure_folder,
            drop_zero_label=False,
        )

if __name__ == "__main__":
    main()