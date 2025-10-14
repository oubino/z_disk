# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit every peak

# %%
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perpl.modelling import zdisk_modelling, zdisk_plots
from perpl.io import plotting

# %% [markdown]
# ## Fit histogram

# %% [markdown]
# ### Params

# %%
fitlength = 150. # standard max distance over which to plot distances and fit models
axial_limit = 200. # This is the YZ-distance limit for X-distances to include
precision = 5.0

numberoflocalisationss = [1,10]
bin_sizes = [5]

# %%
ssrs = []
aics = []
setups = []
for param in list(product(numberoflocalisationss, bin_sizes)):
    numberoflocalisations, bin_size = param

    loc_prec_path = rf"/home/oliver/smlm_cloud/janelia_analysis/experiments/ACTN2/output/perpl_relative_posns/all_z_disks_{precision}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-locprec_150.0filter.txt"
    actn_affimer_relpos_path = rf"/home/oliver/smlm_cloud/janelia_analysis/experiments/ACTN2/output/perpl_relative_posns/all_z_disks_{precision}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-relpos_150.0filter.csv" # path to relative posn data

    loc_precision = np.loadtxt(loc_prec_path) 
    
    relpos = pd.read_csv(actn_affimer_relpos_path)
    relpos = pd.DataFrame({
        "axial": relpos.yy_separation,
        "transverse": relpos.xz_separation},)

    trans_distances = zdisk_modelling.get_transverse_separations(
        relpos,
        max_distance=relpos.transverse.max(),
        axial_limit=axial_limit
        )
    trans_distances = zdisk_modelling.remove_duplicates(trans_distances)
        
    # ## Get the histogram data
    # Up to distance = fitlength
    
    hist_values, bin_edges = zdisk_plots.plot_distance_hist(
        trans_distances,
        fitlength,
        bin_size,
        axial_limit,
        )
    bin_centres = (bin_edges[0:(len(bin_edges) - 1)]
                + bin_edges[1:]
                ) / 2
    
    models = [
        zdisk_modelling.set_up_model_5_peaks_fixed_ratio_with_fit_settings(),
        zdisk_modelling.set_up_model_5_peaks_fixed_ratio_no_replocs_with_fit_settings(),
        zdisk_modelling.set_up_model_5_variable_peaks_with_fit_settings(),
        zdisk_modelling.set_up_model_5_variable_peaks_after_offset_with_fit_settings(),
        zdisk_modelling.set_up_model_4_peaks_fixed_ratio_with_fit_settings(),
        #zdisk_modelling.set_up_model_4_variable_peaks_with_fit_settings(), doesn't fit
        zdisk_modelling.set_up_model_4_variable_peaks_after_offset_with_fit_settings(),
        zdisk_modelling.set_up_model_3_peaks_fixed_ratio_with_fit_settings(),
    ]

    for i, trans_model_with_info in enumerate(models):

        print("Number of localisations: ", numberoflocalisations)
        print("Bin size: ", bin_size)
        print("Model : ", i)
    
        # ## Fit model to histogram bin values, at bin centres
        (params_optimised,
        params_covar,
        params_1sd_error,
        ssr,
        aic) = zdisk_modelling.fitmodel_to_hist(
            bin_centres,
            hist_values,
            trans_model_with_info.model_rpd,
            trans_model_with_info.initial_params,
            trans_model_with_info.param_bounds,
            )

        zdisk_plots.plot_distance_hist_and_fit(
            trans_distances,
            fitlength,
            bin_size,
            params_optimised,
            params_covar,
            trans_model_with_info,
            plot_95ci=True,
            n_locs=numberoflocalisations,
        )

        plt.show()

        ssrs.append(ssr)
        aics.append(aic)
        setups.append(f"model:{i}_nlocs_{numberoflocalisations}_binsize_{bin_size}")
        
aics, ssrs, setups = zip(*sorted(zip(aics, ssrs, setups)))

# %%
print("AICS: ", aics)
print("SSRS: ", ssrs)
print("Setups: ", setups)

# %% [markdown]
# ## Fit RPD

# %%
aics = []
ssrs = []
setups = []

for numberoflocalisations in [1,10,15]:

    loc_prec_path = rf"/home/oliver/smlm_cloud/janelia_analysis/experiments/ACTN2/output/perpl_relative_posns/all_z_disks_{precision}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-locprec_150.0filter.txt"
    actn_affimer_relpos_path = rf"/home/oliver/smlm_cloud/janelia_analysis/experiments/ACTN2/output/perpl_relative_posns/all_z_disks_{precision}precisionfilter_{numberoflocalisations}numberoflocalisations_PERPL-relpos_150.0filter.csv" # path to relative posn data

    loc_precision = np.loadtxt(loc_prec_path) 
    
    relpos = pd.read_csv(actn_affimer_relpos_path)
    relpos = pd.DataFrame({
        "axial": relpos.yy_separation,
        "transverse": relpos.xz_separation},)

    trans_distances = zdisk_modelling.get_transverse_separations(
        relpos,
        max_distance=relpos.transverse.max(),
        axial_limit=axial_limit
        )
    trans_distances = zdisk_modelling.remove_duplicates(trans_distances)
 
    calculation_points = np.arange(fitlength + 1.)
    trans_rpd = plotting.estimate_rpd_churchman_1d(
        input_distances=trans_distances,
        calculation_points=calculation_points,
        combined_precision=(np.sqrt(2) * loc_precision)
    )
    models = [
        zdisk_modelling.set_up_model_5_peaks_fixed_ratio_with_fit_settings(),
        zdisk_modelling.set_up_model_5_peaks_fixed_ratio_no_replocs_with_fit_settings(),
        zdisk_modelling.set_up_model_5_variable_peaks_with_fit_settings(),
        zdisk_modelling.set_up_model_5_variable_peaks_after_offset_with_fit_settings(),
        zdisk_modelling.set_up_model_4_peaks_fixed_ratio_with_fit_settings(),
        zdisk_modelling.set_up_model_4_variable_peaks_with_fit_settings(),
        zdisk_modelling.set_up_model_4_variable_peaks_after_offset_with_fit_settings(),
        zdisk_modelling.set_up_model_3_peaks_fixed_ratio_with_fit_settings(),
        
    ]

    trans_rpd = trans_rpd[calculation_points > 0]
    calculation_points = calculation_points[calculation_points > 0]

    for i, trans_model_with_info in enumerate(models):

        print("Number of localisations: ", numberoflocalisations)
        print("Model : ", i)
    
        # ## Fit model to histogram bin values, at bin centres
        (params_optimised,
        params_covar,
        params_1sd_error,
        ssr,
        aic) = zdisk_modelling.fitmodel_to_hist(
            calculation_points,
            trans_rpd,
            trans_model_with_info.model_rpd,
            trans_model_with_info.initial_params,
            trans_model_with_info.param_bounds,
            )

        plt.plot(calculation_points,trans_rpd)
        zdisk_plots.plot_fitted_model(
            calculation_points,
            fitlength,
            params_optimised,
            params_covar,
            trans_model_with_info,
            plot_95ci=False
            )

        plt.show()

        ssrs.append(ssr)
        aics.append(aic)
        setups.append(f"model:{i}_nlocs_{numberoflocalisations}")
        
aics, ssrs, setups = zip(*sorted(zip(aics, ssrs, setups)))

# %%
print("AICS: ", aics)
print("SSRS: ", ssrs)
print("Setups: ", setups)

# %%
