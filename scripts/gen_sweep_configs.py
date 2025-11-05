import argparse
from itertools import product
import os

import yaml

def gen_configs(config_folder, direction):

    # load in configuration
    with open(os.path.join(config_folder, "config.yaml"), "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in params for particular direction
    direction_params = config[direction]

    # load in params list
    fitlengths = direction_params["fitlength"]
    dimension = direction_params["dimension"]
    backgrounds = direction_params["background"]
    n_peaks = direction_params["n_peaks"]
    peak_types = direction_params["peak_type"]
    charac_dists = direction_params["charac_dist"]
    charac_dist_ratios = direction_params["charac_dist_ratio"]
    repeats = direction_params["repeats"]
    offsets = direction_params["offset"]
    normalises = direction_params["normalise"]
    params_initial = direction_params["params_initial"]
    params_lower = direction_params["params_lower"]
    params_upper = direction_params["params_upper"]

    # generate all possible model configurations
    for index, params in enumerate(list(product(
        fitlengths, 
        dimension,
        backgrounds,
        n_peaks,
        peak_types,
        charac_dists,
        charac_dist_ratios,
        repeats,
        offsets,
        normalises,
    ))):
        model_config = {
            "fitlength": params[0],
            "dimension": params[1],
            "background": params[2],
            "n_peaks": params[3],
            "peak_type": params[4],
            "characteristic_distance": params[5],
            "characteristic_distance_ratio": params[6],
            "repeats": params[7],
            "offset": params[8],
            "normalise": params[9],
            "params_initial": params_initial,
            "params_lower": params_lower,
            "params_upper": params_upper,
        }

        # save yaml file
        model_config_save_loc = os.path.join(config_folder, f"{direction}_models/model_{index}.yaml")
        with open(model_config_save_loc, "w") as outfile:
            yaml.dump(model_config, outfile)

def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Generate configuration files for parameter sweep"
    )

    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        type=str,
        help="name of the experiment",
        required=True,
    )

    args = parser.parse_args(argv)

    config_folder = os.path.join("experiments", args.experiment, "perpl_config")

    for f in ["axial_models", "transverse_models"]:
        folder = os.path.join(config_folder, f)
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} folder already exists")
    
    for f in ["axial_models", "transverse_models"]:
        folder = os.path.join(config_folder, f)
        os.makedirs(folder)

    # generate axial and transverse config files
    gen_configs(config_folder, "axial")
    gen_configs(config_folder, "transverse")

if __name__ == "__main__":
    main()
