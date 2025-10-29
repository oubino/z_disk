import argparse
from itertools import product
import os
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import polars as pl
#import seaborn as sns
import yaml

from perpl.modelling import zdisk_modelling, zdisk_plots
from perpl.modelling.modelling_general import PERPLModel

#from perpl.io import plotting
#from perpl.relative_positions import main as calculate_relative_positions
#from perpl.relative_positions import getdistances, get_vectors, save_relative_positions


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Calculate relative positions using PERPL"
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
    output_folder = os.path.join("experiments", args.experiment, "output")

    # load in configuration
    with open(os.path.join(config_folder, "perpl_config"), "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    relpos_filter = config["relpos_filter"]
    axial_direction = config["axial_direction"]
    transverse_direction = config["transverse_direction"]
    transverse_limit = config["transverse_limit"]
    axial_limit = config["axial_limit"]
    loc_precision_filter = config["loc_precision"]
    numberoflocalisations_lst = config["numberoflocalisations"]
    bin_size_lst = config["bin_sizes"]

    # load in axial models
    axial_models = os.listdir(os.path.join(config_folder, "axial_models"))
    print(f"{len(axial_models)} axial models are being tested")

    axial_models_configs = []
    for i, axial_model in enumerate(axial_models):
        with open(os.path.join(axial_models, axial_model), "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            axial_models_configs.append(config)

    # load in transverse models
    

    # +++ FIT AXIAL +++
    
    # --- Fit histogram
    
    for param in list(product(numberoflocalisations_lst, bin_size_lst)):
        numberoflocalisations, bin_size = param

        loc_prec_path = rf"experiments/{args.experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-locprec_{relpos_filter}filter.txt"
        relpos_path = rf"experiments/{args.experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-relpos_{relpos_filter}filter.csv" # path to relative posn data

        loc_precision = np.loadtxt(loc_prec_path) 
        
        # load in relative positions and calculate axial and tranverse repeat distances
        relpos = pd.read_csv(relpos_path)
        
        relpos = pd.DataFrame({
            "axial": relpos[f"{axial_direction}_separation"],
            "transverse": relpos[f"{transverse_direction}_separation"]},)
        
        axial_distances = zdisk_modelling.getaxialseparations_no_smoothing(
            relpos,
            max_distance=relpos.axial.max(),
            transverse_limit=transverse_limit
            )
        axial_distances = zdisk_modelling.remove_duplicates(axial_distances)

        # for each axial model...
        for i, axial_model in enumerate(axial_models):

            axial_model_config = axial_models_configs[i]

            # Get the histogram data up to distance = fitlength
            hist_values, bin_edges = np.histogram(
                axial_distances,
                bins=np.arange(0, axial_model_config["fitlength"] + 1, bin_size)
            )
            bin_centres = (bin_edges[:- 1] + bin_edges[1:]) / 2
            
            perpl_model = PERPLModel(
                dimension=axial_model_config["dimension"],
                background=axial_model_config["background"],
                n_peaks=axial_model_config["n_peaks"],
                peak_type=axial_model_config["peak_type"],
                repeat_distance=axial_model_config["repeat_distance"],
                repeat_distance_ratio=axial_model_config["repeat_distance_ratio"],
                repeats=axial_model_config["repeats"],
                offset=axial_model_config["offset"],
                normalise=axial_model_config["normalise"],
                params_initial=axial_model_config["params_initial"],
                params_lower=axial_model_config["params_lower"],
                params_upper=axial_model_config["params_upper"],
            )

            perpl_model.fit_to_experiment(
                bin_centres,
                hist_values,
            )

            # plot distance hist and fit
            fig = perpl_model.plot_distance_hist_and_fit(
                axial_distances,
                bin_edges,
                bin_centres,
                axial_model_config["fitlength"],
            )
            figname = os.path.join(
                OUTPUTFOLDER, (f"histandfit_{axial_model.rstrip(".yaml")}_nlocs_{numberoflocalisations}_binsize_{binsize}.svg")
            )
            fig.savefig(figname)

            # plot model components

                # this should save the plot as svg somewhere

            # save ssr, aic, aiccorr, setup
    
    # note which is best model 

    # save the results of which is best model to .txt file

    # --- Fit KDE

    # for n_locs

    # +++ FIT TRANSVERSE +++

    # --- Fit histogram

    # for n_locs, bin_size ...

    # --- Fit KDE

    # for n_locs

    


if __name__ == "__main__":
    main()
