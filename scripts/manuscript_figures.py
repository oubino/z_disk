# Prepare figures for manuscript
import os
import polars as pl

## Best models for each protein
best_models = []

#fitlengths = {
#    "ACTN2_axial": 115,
#    "ACTN2_transverse": 57,
#    "Z1Z2_axial": 110,
#    "Z1Z2_transverse": 55,
#    "ZASP6_axial": 115,
#    "ZASP6_transverse": 47,
#}

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

        #x = x.filter(pl.col("Fitlength") == fitlengths[f"{protein}_{direction}"])

        best_models.append(x)

best_models = pl.concat(best_models)
with pl.Config() as cfg:

    cfg.set_tbl_cols(10)
    cfg.set_tbl_rows(14)
    print(best_models)

## Simpler models for each protein



