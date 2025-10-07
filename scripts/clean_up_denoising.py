"""Clean up after denoising module

Module cleans up after the denoising step
"""

import argparse
import os
import shutil


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If try to convert but already files there"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Clean up after denoising")

    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        type=str,
        help="name of the experiment",
        required=True,
    )

    args = parser.parse_args(argv)

    input_folder = os.path.join(
        "experiments", args.experiment, "output/segmented_z_disks"
    )
    output_folder = os.path.join(
        "experiments", args.experiment, "output/segmented_z_disks_denoised"
    )

    if not os.path.exists(output_folder):
        raise ValueError("No denoising data")

    files = os.listdir(input_folder)
    denoised_files = os.listdir(output_folder)
    denoised_files_compare = [f.replace("_denoised", "") for f in denoised_files]

    # Check that all files in denoising folder also in input folder
    assert len([f for f in denoised_files_compare if f not in files]) == 0

    # Print out files in denoising folder and in input folder
    print("Files that have been denoised: ", denoised_files)

    # Check which files are in input folder but not denoising folder
    not_denoised = [f for f in files if f not in denoised_files_compare]
    print("Files that have not been denoised: ", not_denoised)

    # Copy these files across
    copy_ = input("Copy these not denoised files across? (Y) ")
    if copy_ == "Y":
        for file in not_denoised:
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, file)

            shutil.copyfile(input_file, output_file)


if __name__ == "__main__":
    main()
