import shutil

output_folder = "manuscript_figures/figure_g1"

models = {
    "ACTN2_axial": "model_39_locprec_6.0_nlocs_1_fitlength_125",
    "ACTN2_transverse": "model_69_locprec_6.0_nlocs_1_fitlength_62",
    "Z1Z2_axial": "model_28_locprec_8.0_nlocs_1_fitlength_120",
    "Z1Z2_transverse": "model_70_locprec_7.0_nlocs_1_fitlength_60",
    "ZASP6_axial": "model_16_locprec_9.0_nlocs_1_fitlength_120",
    "ZASP6_transverse": "model_60_locprec_9.0_nlocs_1_fitlength_52",
}

for key in models.keys():

    protein, direction = key.split("_")

    model = models[key]

    print(protein, " ", direction)

    input_file = f"experiments/{protein}/output/perpl_modelling/{direction}/kdes/{model}_kdeandfit.svg"

    src = input_file
    dest = output_folder + f"/{protein}_{direction}.svg"

    print(src)
    print(dest)

    shutil.copy(src, dest)



