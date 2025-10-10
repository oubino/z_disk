import argparse
import os
import shutil

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
        "-i",
        "--files_to_include",
        action="store",
        type=str,
        help="files visualised and checked",
        required=True,
        nargs="+",
    )

    args = parser.parse_args(argv)

    folder = os.path.join("experiments", args.experiment, "output")

    input_folder = os.path.join(folder, "segmented_z_disks_denoised_filtered")
    output_folder = os.path.join(folder, "segmented_z_disks_denoised_filtered_vischecked")

    # input folder
    loc_prec_folders = [
        os.path.join(input_folder, x)
        for x in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, x))
    ]
    if len(loc_prec_folders) == 0:
        raise ValueError("No input folders")
    
    # output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Files being included: ", args.files_to_include)

    for input_folder in loc_prec_folders:

        localisation_precision_filter = input_folder.split("/")[-1].replace(
            "nm_filter", ""
        )

        output_sub_folder = os.path.join(output_folder, f"{localisation_precision_filter}nm_filter")

        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)

        input_files = os.listdir(input_folder)

        files_to_copy = [x for x in input_files if x in args.files_to_include]

        for file in files_to_copy:
            src = os.path.join(input_folder, file)
            dest = os.path.join(output_sub_folder, file)
            shutil.copyfile(src, dest)

if __name__ == "__main__":
    main()