import argparse
from itertools import product
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
#import seaborn as sns
import yaml

from perpl.io import plotting
from perpl.modelling import zdisk_modelling
from perpl.modelling.modelling_general import PERPLModel

def model_the_data(direction,
                   plot_type,
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
                   fitlengths,
                   nlocs,
                   bgbelowzeros,
                   ):
            
    models = models[direction]

    loc_prec_path = rf"experiments/{experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-locprec_{relpos_filter}filter.txt"
    relpos_path = rf"experiments/{experiment}/output/perpl_relative_posns/all_z_disks_{loc_precision_filter}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-relpos_{relpos_filter}filter.csv" # path to relative posn data

    loc_precision = np.loadtxt(loc_prec_path) 
    
    # load in relative positions and calculate axial and transverse characteristic distances
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

        if plot_type == "histogram":

            # Get the histogram data up to distance = fitlength
            hist_values, bin_edges = np.histogram(
                distances,
                bins=np.arange(0, model_config["fitlength"] + 1, bin_size)
            )
            bin_centres = (bin_edges[:- 1] + bin_edges[1:]) / 2

            x_expt = bin_centres
            y_expt = hist_values

        elif plot_type == "kde":

            if len(distances) == 0:
                print(f"Skipping {model_name} as no distances to fit")
                continue

            increment = np.round(model_config["fitlength"]/len(distances))
            if increment == 0:
                increment = 1
            calculation_points = np.arange(0, model_config["fitlength"] + 1., increment)
            
            if model_config["dimension"] == 1:
                churchman = plotting.estimate_rpd_churchman_1d
            elif model_config["dimension"] == 2:
                churchman = plotting.estimate_rpd_churchman_2d
            elif model_config["dimension"] == 3:
                churchman = plotting.estimate_rpd_churchman_3d

            rpd = churchman(
                input_distances=distances,
                calculation_points=calculation_points,
                combined_precision=(np.sqrt(2) * loc_precision)
            )
            
            y_expt = rpd[calculation_points > 0]
            x_expt = calculation_points[calculation_points > 0]

        perpl_model = PERPLModel(
            dimension=model_config["dimension"],
            background=model_config["background"],
            n_peaks=model_config["n_peaks"],
            peak_type=model_config["peak_type"],
            characteristic_distance=model_config["characteristic_distance"],
            characteristic_distance_ratio=model_config["characteristic_distance_ratio"],
            repeats=model_config["repeats"],
            offset=model_config["offset"],
            normalise=model_config["normalise"],
            params_initial=model_config["params_initial"],
            params_lower=model_config["params_lower"],
            params_upper=model_config["params_upper"],
            name=model_name,
        )

        if model_config["background"] is None and model_config["n_peaks"] == 0 and model_config["repeats"] is False:
            print(f"Skipping {model_name} as has nothing to fit")
            continue
        
        # print("Model name ", model_name) Debug

        perpl_model.fit_to_experiment(
            x_expt,
            y_expt, 
        )
        
        if plot_type == "histogram":
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

        elif plot_type == "kde":
            # plot kde and fit
            fig = perpl_model.plot_distance_kde_and_fit(
                x_expt,
                y_expt,
                model_config["fitlength"]
            )
            figname = os.path.join(
                output_folder, 
                "kdes",
                (f"{model_name}_nlocs_{numberoflocalisations}_kdeandfit.svg")
            )

        if fig is not None:
            fig.savefig(figname)
            plt.close(fig)

        # plot model components
        fig2 = perpl_model.plot_model_components(
            model_config["fitlength"]
        )
        if plot_type == "histogram":
            figname = os.path.join(
                output_folder, 
                "histograms",
                (f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}_modelcomponents.svg")
            )
        elif plot_type == "kde":
            figname = os.path.join(
                output_folder, 
                "kdes",
                (f"{model_name}_nlocs_{numberoflocalisations}_modelcomponents.svg")
            )
        if fig2 is not None:
            fig2.savefig(figname)
            plt.close(fig2)

        # save model params and err
        if plot_type == "histogram":
            opt_param_path = os.path.join(output_folder, 
                                        "histograms", 
                                        f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}_optparams.txt")
        elif plot_type == "kde":
            opt_param_path = os.path.join(output_folder, 
                                        "kdes", 
                                        f"{model_name}_nlocs_{numberoflocalisations}_optparams.txt")
        with open(opt_param_path, "w") as f:
            f.write("Optimal params +- Error\n")
            f.write("-----------------------\n")
            if perpl_model.params_optimised is None:
                f.write("Model failed to fit")
            else:
                for row in zip(perpl_model.param_names, perpl_model.params_optimised, perpl_model.params_err):
                    f.write(f"{row[0]}: {row[1]} +- {row[2]}\n")

        # save ssr, aic, aiccorr, setup
        ssrs.append(perpl_model.sum_of_squares_error)
        aics.append(perpl_model.aic)
        aiccorrs.append(perpl_model.aic_corrected)
        if plot_type == "histogram":
            setups.append(f"{model_name}_nlocs_{numberoflocalisations}_binsize_{bin_size}")
        elif plot_type == "kde":
            setups.append(f"{model_name}_nlocs_{numberoflocalisations}")
        fitlengths.append(model_config["fitlength"])
        nlocs.append(numberoflocalisations)
        bgbelowzeros.append(perpl_model.bgbelowzero)

def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Model the data using PERPL"
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
        "-fh",
        "--fit_histograms",
        action="store_true",
        help="fit histograms",
        required=False,
    )

    args = parser.parse_args(argv)

    config_folder = os.path.join("experiments", args.experiment, "perpl_config")

    output_modelling_folder = os.path.join("experiments", args.experiment, "output/perpl_modelling")

    if not os.path.exists(output_modelling_folder):
        os.makedirs(output_modelling_folder)

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
    transverse_models = os.listdir(os.path.join(config_folder, "transverse_models"))
    print(f"{len(transverse_models)} transverse models are being tested")

    transverse_models_configs = []
    for i, transverse_model in enumerate(transverse_models):
        with open(os.path.join(config_folder, "transverse_models", transverse_model), "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            transverse_models_configs.append(config)

    # one list
    models = {
        "axial": axial_models,
        "transverse": transverse_models,
    }

    model_configs ={
        "axial": axial_models_configs,
        "transverse": transverse_models_configs,
    }

    
    # +++ FIT AXIAL....

    output_folder = os.path.join(output_modelling_folder, "axial")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for f in ["histograms", "kdes"]:
        i = os.path.join(output_folder, f)
        if not os.path.exists(i):
            os.makedirs(i)

    # .... histogram

    if args.fit_histograms:

        ssrs = []
        aics = []
        aiccorrs = []
        setups = []
        fitlengths = []
        nlocs = []
        bgbelowzeros = []

        for param in list(product(numberoflocalisations_lst, bin_size_lst)):
            numberoflocalisations, bin_size = param

            model_the_data(
                "axial",
                "histogram",
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
                fitlengths,
                nlocs,
                bgbelowzeros,
            )

        aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros, = zip(*sorted(zip(aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros)))
        
        with open(os.path.join(output_folder, "results_histograms.csv"), "w") as f:
            f.write("Model,AICcorr,AIC,SSR,Fitlength,Nlocs,BGbelowzero\n")
            for row in zip(setups, aiccorrs, aics, ssrs, fitlengths, nlocs, bgbelowzeros):
                f.write(",".join(map(str, row)) + "\n")

    # ... KDE

    ssrs = []
    aics = []
    aiccorrs = []
    setups = []
    fitlengths = []
    nlocs = []
    bgbelowzeros = []

    for numberoflocalisations in numberoflocalisations_lst:

        model_the_data(
            "axial",
            "kde",
            limits,
            models,
            model_configs,
            args.experiment,
            loc_precision_filter,
            None,
            numberoflocalisations,
            relpos_filter,
            axial_direction,
            transverse_direction,
            output_folder,
            ssrs,
            aics,
            aiccorrs,
            setups,
            fitlengths,
            nlocs,
            bgbelowzeros,
        )

    aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros, = zip(*sorted(zip(aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros)))
    
    with open(os.path.join(output_folder, "results_kdes.csv"), "w") as f:
        f.write("Model,AICcorr,AIC,SSR,Fitlength,Nlocs,BGbelowzero\n")
        for row in zip(setups, aiccorrs, aics, ssrs, fitlengths, nlocs, bgbelowzeros):
            f.write(",".join(map(str, row)) + "\n")

    # +++ FIT TRANSVERSE +++

    output_folder = os.path.join(output_modelling_folder, "transverse")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for f in ["histograms", "kdes"]:
        i = os.path.join(output_folder, f)
        if not os.path.exists(i):
            os.makedirs(i)
    
    # .... histogram

    if args.fit_histograms:

        ssrs = []
        aics = []
        aiccorrs = []
        setups = []
        fitlengths = []
        nlocs = []
        bgbelowzeros = []

        for param in list(product(numberoflocalisations_lst, bin_size_lst)):
            numberoflocalisations, bin_size = param

            model_the_data(
                "transverse",
                "histogram",
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
                fitlengths,
                nlocs,
                bgbelowzeros,
            )

        aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros, = zip(*sorted(zip(aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros)))
        
        with open(os.path.join(output_folder, "results_histograms.csv"), "w") as f:
            f.write("Model,AICcorr,AIC,SSR,Fitlength,Nlocs,BGbelowzero\n")
            for row in zip(setups, aiccorrs, aics, ssrs, fitlengths, nlocs, bgbelowzeros):
                f.write(",".join(map(str, row)) + "\n")

    # ... KDE

    ssrs = []
    aics = []
    aiccorrs = []
    setups = []
    fitlengths = []
    nlocs = []
    bgbelowzeros = []


    for numberoflocalisations in numberoflocalisations_lst:

        model_the_data(
            "transverse",
            "kde",
            limits,
            models,
            model_configs,
            args.experiment,
            loc_precision_filter,
            None,
            numberoflocalisations,
            relpos_filter,
            axial_direction,
            transverse_direction,
            output_folder,
            ssrs,
            aics,
            aiccorrs,
            setups,
            fitlengths,
            nlocs,
            bgbelowzeros,
        )

    aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros, = zip(*sorted(zip(aiccorrs, aics, ssrs, setups, fitlengths, nlocs, bgbelowzeros)))
    
    with open(os.path.join(output_folder, "results_kdes.csv"), "w") as f:
        f.write("Model,AICcorr,AIC,SSR,Fitlength,Nlocs,BGbelowzero\n")
        for row in zip(setups, aiccorrs, aics, ssrs, fitlengths, nlocs, bgbelowzeros):
            f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
