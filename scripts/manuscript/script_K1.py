# Generate figure k1 and table k1

import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_folder = "manuscript_figures"

table_protein = []
table_xprec_med = []
table_yprec_med = []
table_zprec_med = []
table_nzdisks = []
table_locsperzdisk_med = []
table_zdisksize_x_med = []
table_zdisksize_y_med = []
table_zdisksize_z_med = []

for protein in ["ACTN2", "Z1Z2", "ZASP6"]:

    #print(f"----- Protein: {protein} ------ ")

    input_folder = os.path.join(f"experiments/{protein}/output/segmented_z_disks_denoised")

    files = os.listdir(input_folder)

    # calculate localisation precision before filtering

    x_prec = []
    y_prec = []
    z_prec = []
    locs_per_zdisk = []
    zdisksize_x = []
    zdisksize_y = []
    zdisksize_z = []

    for file in files:

        file_path = os.path.join(input_folder, file)

        df = pl.read_csv(file_path)

        x_prec.append(df["Group Sigma X Pos"])
        y_prec.append(df["Group Sigma Y Pos"])
        z_prec.append(df["Group Sigma Z"])

        locs_per_zdisk.append(len(df))

        zdisksize_x.append(df["x"].max() - df["x"].min())
        zdisksize_y.append(df["y"].max() - df["y"].min())
        zdisksize_z.append(df["z"].max() - df["z"].min())


    x_prec = np.concatenate(x_prec)
    y_prec = np.concatenate(y_prec)
    z_prec = np.concatenate(z_prec)

    x_prec = [float(x.lstrip(" ")) for x in x_prec]
    y_prec = [float(y.lstrip(" ")) for y in y_prec]
    z_prec = [float(z.lstrip(" ")) for z in z_prec]

    pd_df = pd.DataFrame(
        {
            "X (nm)": x_prec,
            "Y (nm)": y_prec,
            "Z (nm)": z_prec,
        }
    )

    # Use Arial font and set global font sizes
    plt.rcParams.update(
        {
            "axes.titlesize": 16,  # title size
            "axes.labelsize": 14,  # x/y label size
            "xtick.labelsize": 12,  # x tick label size
            "ytick.labelsize": 12,  # y tick label size
            "legend.fontsize": 12,  # legend label size
        }
    )

    plt.figure(figsize=(10, 6))
    plot = sns.histplot(
        data=pd_df,
        element="step",
        stat="count",
    )

    legend = plot.get_legend()
    legend.set_title(None)
    plt.xlim(0, 40)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.xlabel("Localisation precision")
    plt.ylabel("Localisation count")
    plt.title(f"Localisation precision for {protein} (pre-filtering)")
    output_loc = os.path.join(
        output_folder, f"figure_k1/{protein}.svg"
    )
    plt.savefig(output_loc, bbox_inches="tight", transparent=True)

    # generate table k1
    table_protein.append(protein)
    table_xprec_med.append(pd_df.median()["X (nm)"].round(1))
    table_yprec_med.append(pd_df.median()["Y (nm)"].round(1))
    table_zprec_med.append(pd_df.median()["Z (nm)"].round(1))
    table_nzdisks.append(len(files))
    table_locsperzdisk_med.append(np.median(locs_per_zdisk))
    table_zdisksize_x_med.append(np.median(zdisksize_x).round(1))
    table_zdisksize_y_med.append(np.median(zdisksize_y).round(1))
    table_zdisksize_z_med.append(np.median(zdisksize_z).round(1))

    #print("Mean precision (pre-filtering)")
    #print(pd_df.mean())

    #print("Median precision (pre-filtering)")
    #print(pd_df.median())

df = pd.DataFrame(
    {"Protein": table_protein,
     "N z disks": table_nzdisks,
     "Med z disk size in x": table_zdisksize_x_med,
     "Med z disk size in y": table_zdisksize_y_med,
     "Med z disk size in z": table_zdisksize_z_med,
     "Med locs per z disk": table_locsperzdisk_med,
     "Median x precision": table_xprec_med,
     "Median y precision": table_yprec_med,
     "Median z precision": table_zprec_med,
     }  
)

output_loc = os.path.join(
        output_folder, f"table_k1.csv"
    )
df.to_csv(output_loc, index=False)
