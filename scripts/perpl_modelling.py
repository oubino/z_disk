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


def model_the_data(direction, 
                   limits, 
                   models,
                   model_configs,
                   experiment,
                   loc_precision_filter,
                   bin_size,
                   numberoflocalisations,
                   relpos_filter,
                   axial_direction,
                   transverse_direction,
                   output_folder,
                   ssrs,
                   aics,
                   aiccorrs,
                   setups,
                   ):
            
    models = models[direction]

    loc_prec_path = rf"experiments/{experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-locprec_{relpos_filter}filter.txt"
    relpos_path = rf"experiments/{experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-relpos_{relpos_filter}filter.csv" # path to relative posn data

    loc_precision = np.loadtxt(loc_prec_path) 
    
    # load in relative positions and calculate axial and tranverse repeat distances
    relpos = pd.read_csv(relpos_path)
    
    relpos = pd.DataFrame({
        "axial": relpos[f"{axial_direction}_separation"],
        "transverse": relpos[f"{transverse_direction}_separation"]},)
    
    if direction == "axial":
        distances = zdisk_modelling.getaxialseparations_no_smoothing(
            relpos,
            max_distance=relpos[direction].max(),
            transverse_limit=limits["transverse"]
            )
    elif direction == "transverse":
        distances = zdisk_modelling.get_transverse_separations(
            relpos,
            max_distance=relpos[direction].max(),
            axial_limit=limits["axial"]
            )
    
    distances = zdisk_modelling.remove_duplicates(distances)

    # for each model...
    for i, model in enumerate(models):
        
        model_name = model.rstrip(".yaml")
        model_config = model_configs[direction][i]

        # Get the histogram data up to distance = fitlength
        hist_values, bin_edges = np.histogram(
            distances,
            bins=np.arange(0, model_config["fitlength"] + 1, bin_size)
        )
        bin_centres = (bin_edges[:- 1] + bin_edges[1:]) / 2
        
        perpl_model = PERPLModel(
            dimension=model_config["dimension"],
            background=model_config["background"],
            n_peaks=model_config["n_peaks"],
            peak_type=model_config["peak_type"],
            repeat_distance=model_config["repeat_distance"],
            repeat_distance_ratio=model_config["repeat_distance_ratio"],
            repeats=model_config["repeats"],
            offset=model_config["offset"],
            normalise=model_config["normalise"],
            params_initial=model_config["params_initial"],
            params_lower=model_config["params_lower"],
            params_upper=model_config["params_upper"],
        )

        perpl_model.fit_to_experiment(
            bin_centres,
            hist_values,
        )

        # plot distance hist and fit
        fig = perpl_model.plot_distance_hist_and_fit(
            distances,
            bin_edges,
            bin_centres,
            model_config["fitlength"],
        )
        figname = os.path.join(
            output_folder, 
            "histograms",
            (f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}_histandfit.svg")
        )
        fig.savefig(figname)

        # plot model components
        fig = perpl_model.plot_model_components(
            model_config["fitlength"]
        )
        figname = os.path.join(
            output_folder, 
            "histograms",
            (f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}_modelcomponents.svg")
        )
        fig.savefig(figname)

        # save model params and err
        with open(os.path.join(output_folder, "histograms", f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}_optparams.txt"), "w") as f:
            f.write("Optimal params +- Error\n")
            f.write("-----------------------\n")
            for row in zip(perpl_model.param_names, perpl_model.params_optimised, perpl_model.params_err):
                f.write(f"{row[0]}: {row[1]} +- {row[2]}\n")

        # save ssr, aic, aiccorr, setup
        ssrs.append(perpl_model.sum_of_squares_error)
        aics.append(perpl_model.aic)
        aiccorrs.append(perpl_model.aic_corrected)
        setups.append(f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}")

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

    output_folder = os.path.join("experiments", args.experiment, "output/perpl_modelling")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load in configuration
    with open(os.path.join(config_folder, "config.yaml"), "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    relpos_filter = config["relpos_filter"]
    axial_direction = config["axial_direction"]
    transverse_direction = config["transverse_direction"]
    transverse_limit = config["transverse_limit"]
    axial_limit = config["axial_limit"]
    limits={
        "transverse": transverse_limit,
        "axial": axial_limit,
    }
    loc_precision_filter = config["loc_precision_filter"]
    numberoflocalisations_lst = config["numberoflocalisations"]
    bin_size_lst = config["bin_sizes"]

    # load in axial models
    axial_models = os.listdir(os.path.join(config_folder, "axial_models"))
    print(f"{len(axial_models)} axial models are being tested")

    axial_models_configs = []
    for i, axial_model in enumerate(axial_models):
        with open(os.path.join(config_folder, "axial_models", axial_model), "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            axial_models_configs.append(config)

    # load in transverse models


    # one list
    models = {
        "axial": axial_models,
        #"transvserse": transverse_models,
    }

    model_configs ={
        "axial": axial_models_configs,
        #"transverse": transverse_models_configs,
    }

    
    # +++ FIT AXIAL....

    output_folder = os.path.join(output_folder, "axial")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for f in ["histograms", "kdes"]:
        i = os.path.join(output_folder, f)
        if not os.path.exists(i):
            os.makedirs(i)
    
    # .... histogram

    ssrs = []
    aics = []
    aiccorrs = []
    setups = []

    for param in list(product(numberoflocalisations_lst, bin_size_lst)):
        numberoflocalisations, bin_size = param

        model_the_data(
            "axial",
            limits,
            models,
            model_configs,
            args.experiment,
            loc_precision_filter,
            bin_size,
            numberoflocalisations,
            relpos_filter,
            axial_direction,
            transverse_direction,
            output_folder,
            ssrs,
            aics,
            aiccorrs,
            setups,
        )

    aiccorrs, aics, ssrs, setups = zip(*sorted(zip(aiccorrs, aics, ssrs, setups)))
    
    with open(os.path.join(output_folder, "results.csv"), "w") as f:
        f.write("Model,AICcorr,AIC,SSR\n")
        for row in zip(setups, aiccorrs, aics, ssrs):
            f.write(",".join(map(str, row)) + "\n")

    # ... KDE



    # for n_locs

    # +++ FIT TRANSVERSE +++

    # --- Fit histogram

    # for n_locs, bin_size ...

    # --- Fit KDE

    # for n_locs


if __name__ == "__main__":
    main()
