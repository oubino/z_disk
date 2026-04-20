# Prepare figures for manuscript
import cairosvg
import os
import numpy as np
import polars as pl
import yaml
import xlsxwriter
from string import ascii_uppercase

## Best models for each protein
best_models = {}

for protein in ["ACTN2", "Z1Z2", "ZASP6"]:

    protein_folder = f"experiments/{protein}/output/perpl_modelling"

    for direction in ["axial", "transverse"]:

        results_file = os.path.join(protein_folder, direction, "results_kdes.csv")

        results_file = pl.read_csv(results_file)

        if "POptAtBounds" in results_file.columns:
            poptatbounds_col_name = "POptAtBounds"
        elif " POptAtBounds" in results_file.columns:
            poptatbounds_col_name = " POptAtBounds"

        if "LargeUncertainty" in results_file.columns:
            largeuncertainty_col_name = "LargeUncertainty"
        elif " LargeUncertainty" in results_file.columns:
            largeuncertainty_col_name = " LargeUncertainty"

        results_file_top = results_file.filter(
            pl.col(largeuncertainty_col_name) == False,
            pl.col(poptatbounds_col_name) == False,
            pl.col("BGbelowzero") == False,
        ).drop(
            pl.col(largeuncertainty_col_name),
            pl.col(poptatbounds_col_name),
            pl.col("BGbelowzero"),
        )


        x = results_file_top.group_by("Locprecision", "Fitlength", "Nlocs").agg(
            pl.col("AICcorr").min().alias("MinAICcorr")
        )
        # pick out best models within 0.01 relative likelihood pp77-78 Burnham
        # Model selection and multimodel inference
        x = x.join(
           results_file_top,
            on=["Locprecision", "Fitlength", "Nlocs"],
            how="right",
            ).filter(
                pl.col("AICcorr") <= pl.col("MinAICcorr") + 9.210340372
                ).sort(
                    pl.col("Locprecision", "Fitlength", "Nlocs")
                    )
        reordered_cols = ["Locprecision", "Fitlength", "Nlocs", "AICcorr"]
        remaining_cols = [c for c in x.columns if c not in reordered_cols]
        reordered_cols.extend(remaining_cols)
        x = x.select(reordered_cols)

        x = x.insert_column(0, pl.Series("Direction", [direction] * len(x)))
        x = x.insert_column(0, pl.Series("Protein", [protein] * len(x)))

        # --- load in the configuration for the model ---
        perpl_model_config_folder = (
            f"experiments/{protein}/perpl_config/{direction}_models"
        )
        bg = []
        n_peaks = []
        peak_type = []
        charac_dist = []
        repeats = []

        for model in x["Model"]:
            model = os.path.join(
                perpl_model_config_folder, "_".join(model.split("_")[:2]) + ".yaml"
            )
            # load in configuration
            with open(model, "r") as ymlfile:
                model = yaml.safe_load(ymlfile)
            bg.append(model["background"])
            n_peaks.append(model["n_peaks"])
            peak_type.append(model["peak_type"])
            charac_dist.append(model["characteristic_distance"])
            repeats.append(model["repeats"])

        x = x.insert_column(len(x.columns), pl.Series("Background", bg))
        x = x.insert_column(len(x.columns), pl.Series("N peaks", n_peaks))
        x = x.insert_column(len(x.columns), pl.Series("Peak type", peak_type))
        x = x.insert_column(len(x.columns), pl.Series("Charac. dist", charac_dist))
        x = x.insert_column(len(x.columns), pl.Series("Repeats", repeats))

        # --- load in the optimal parameters for the model ----
        opt_param_dicts = []
        global_keys = []
        for model in x["Model"]:
            opt_param_loc = f"experiments/{protein}/output/perpl_modelling/{direction}/kdes/{model}_optparams.txt"
            with open(opt_param_loc, "r") as txt_file:
                txt_file = txt_file.read().split("\n")[2:-1]

                keys = []
                values = []
                for param in txt_file:
                    param = param.split(":")
                    keys.append(param[0])
                    val = param[1].lstrip(" ").split("+-")
                    values.append(
                        "+-".join(
                            [
                                str(np.round(float(val[0]), 2)),
                                str(np.round(float(val[1]), 2)),
                            ]
                        )
                    )

                opt_param_dicts.append(dict(zip(keys, values)))
                global_keys.extend(keys)

        global_keys = sorted(set(global_keys))
        global_dict = {k: [] for k in global_keys}

        for opt_param_dict in opt_param_dicts:
            for key in global_keys:
                if key in opt_param_dict.keys():
                    global_dict[key].append(opt_param_dict[key])
                else:
                    global_dict[key].append(None)

        # add optimal parameters into dataframe
        for key, value in global_dict.items():
            x = x.insert_column(len(x.columns), pl.Series(key, value))

        # calculate S.D. of the residuals
        ## pp81 of INTRODUCTION TO LINEAR REGRESSION ANALYSIS, Montgomery, Peck, Vining 
        x = x.with_columns(
            (pl.col("SSR")/(pl.col("Ndatapoints") - pl.col("Nparams")))
            .sqrt()
            .alias("SD of residuals (corrected by no. of params)")
        )

        x = x.with_columns(
            (pl.col("SSR")/(pl.col("Ndatapoints")))
            .sqrt()
            .alias("SD of residuals")
        )

        best_models[f"{protein}_{direction}"] = x

# save each protein + direction to a new sheet in .xlsx workbook
output_excel = "manuscript_figures/table_g1.xlsx"
with xlsxwriter.Workbook(output_excel) as workbook:

    for protein in ["ACTN2", "Z1Z2", "ZASP6"]:
        for direction in ["axial", "transverse"]:

            df = best_models[f"{protein}_{direction}"]

            df.write_excel(workbook=workbook, worksheet=f"{protein}_{direction}")

            # add the image for the model components to the worksheet

            models = df["Model"]
            worksheet = workbook.get_worksheet_by_name(f"{protein}_{direction}")
            worksheet.set_column(len(df.columns), len(df.columns), 200)

            for i, model in enumerate(models):

                model_loc = f"experiments/{protein}/output/perpl_modelling/{direction}/kdes/{model}_kdeandfit.svg"

                # convert svg to png
                png_loc = f"manuscript_figures/table_g1/{protein}_{direction}_{model}_kdeandfit.png"
                cairosvg.svg2png(url=model_loc, write_to=png_loc)

                # embed the image in the worksheet
                try:
                    worksheet.set_row(i + 1, 100)  # Set the row height
                    worksheet.embed_image(i + 1, len(df.columns), png_loc)
                except:
                    column = ascii_uppercase[len(df.columns)]
                    worksheet.insert_image(
                        f"{column}{i+1}", png_loc, {"x_scale": 0.1, "y_scale": 0.1}
                    )

## Simpler models for each protein
