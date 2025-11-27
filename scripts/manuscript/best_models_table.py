# Prepare figures for manuscript
import cairosvg
import os
import polars as pl
import yaml
import xlsxwriter
from string import ascii_uppercase

## Best models for each protein
best_models = []

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

        x = results_file_top["Locprecision", "Fitlength", "Nlocs"].unique()

        x = results_file_top.group_by(
            "Locprecision", "Fitlength", "Nlocs"
        ).agg(pl.col("AICcorr").min())

        x = x.join(results_file_top, on=["Locprecision", "Fitlength", "Nlocs", "AICcorr"], how="left").sort(
            pl.col("Locprecision", "Fitlength", "Nlocs")
        )

        x = x.insert_column(0, pl.Series("Direction", [direction]*len(x)))
        x = x.insert_column(0, pl.Series("Protein", [protein]*len(x)))

        # --- load in the parameters for the model ---
        perpl_model_config_folder = f"experiments/{protein}/perpl_config/{direction}_models"
        bg = []
        n_peaks = []
        peak_type = []
        charac_dist = []
        repeats = []

        for model in x["Model"]:
            model = os.path.join(perpl_model_config_folder, "_".join(model.split("_")[:2]) + ".yaml")
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

        best_models.append(x)

best_models = pl.concat(best_models)

# save each protein + direction to a new sheet in .xlsx workbook
output_excel = "manuscript_figures/table_g1.xlsx"
with xlsxwriter.Workbook(output_excel) as workbook:

    for protein in ["ACTN2", "Z1Z2", "ZASP6"]:
        for direction in ["axial", "transverse"]:

            df = best_models.filter(
                pl.col("Protein") == protein,
                pl.col("Direction") == direction,
            )

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
                    worksheet.set_row(i+1, 100)  # Set the row height       
                    worksheet.embed_image(i+1, len(df.columns), png_loc)
                except:
                    column = ascii_uppercase[len(df.columns)]
                    worksheet.insert_image(f"{column}{i+1}", png_loc, {"x_scale": 0.1, "y_scale": 0.1})

## Simpler models for each protein

