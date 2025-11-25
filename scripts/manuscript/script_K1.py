# Generate figure k1 and table k1

import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_folder = "manuscript_figures"

table_protein = []
table_xprec_mean = []
table_yprec_mean = []
table_zprec_mean = []
table_nzdisks = []
table_locsperzdisk_mean = []
table_zdisksize_x_mean = []
table_zdisksize_y_mean = []
table_zdisksize_z_mean = []

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
            "x": x_prec,
            "y": y_prec,
            "z": z_prec,
        }
    )

    # Use Arial font and set global font sizes
    plt.rcParams.update(
        {
            "axes.titlesize": 12,  # title size
            "axes.labelsize": 11,  # x/y label size
            "xtick.labelsize": 9,  # x tick label size
            "ytick.labelsize": 9,  # y tick label size
            "legend.fontsize": 11,  # legend label size
        }
    )

    plt.figure(figsize=(4, 2.4)) # 10, 6
    plot = sns.histplot(
        data=pd_df,
        element="step",
        stat="count",
    )

    legend = plot.get_legend()
    legend.set_title(None)
    plt.xlim(0, 40)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.xlabel("Localisation precision / nm")
    plt.ylabel("Localisation count")
    #plt.title(f"Localisation precision for {protein} (pre-filtering)")
    plt.title(f"{protein}")
    output_loc = os.path.join(
        output_folder, f"figure_k1/{protein}.svg"
    )
    plt.savefig(output_loc, bbox_inches="tight", transparent=True)

    # generate table k1
    table_protein.append(protein)

    xprec_mean = pd_df.mean()["x"].round(1)
    xprec_std = pd_df.std(ddof=0)["x"].round(1)
    table_xprec_mean.append(f"{xprec_mean} " +  u"\u00B1" + f" {xprec_std}")

    yprec_mean = pd_df.mean()["y"].round(1)
    yprec_std = pd_df.std(ddof=0)["y"].round(1)
    table_yprec_mean.append(f"{yprec_mean} " +  u"\u00B1" + f" {yprec_std}")

    zprec_mean = pd_df.mean()["z"].round(1)
    zprec_std = pd_df.std(ddof=0)["z"].round(1)
    table_zprec_mean.append(f"{zprec_mean} " +  u"\u00B1" + f" {zprec_std}")

    table_nzdisks.append(len(files))

    table_locsperzdisk_mean.append(f"{int(np.mean(locs_per_zdisk).round(0))} " + u"\u00B1" + f" {int(np.std(locs_per_zdisk, ddof=0).round(0))}")

    table_zdisksize_x_mean.append(f"{np.mean(zdisksize_x).round(1)} " + u"\u00B1" + f" {np.std(zdisksize_x, ddof=0).round(1)}")

    table_zdisksize_y_mean.append(f"{np.mean(zdisksize_y).round(1)} " + u"\u00B1" + f" {np.std(zdisksize_y, ddof=0).round(1)}")

    table_zdisksize_z_mean.append(f"{np.mean(zdisksize_z).round(1)} " + u"\u00B1" + f" {np.std(zdisksize_z, ddof=0).round(1)}")

    #print("Mean precision (pre-filtering)")
    #print(pd_df.mean())

    #print("Median precision (pre-filtering)")
    #print(pd_df.median())

df = pd.DataFrame(
    {"Protein": table_protein,
     "N z disks": table_nzdisks,
     "mean z disk size in x": table_zdisksize_x_mean,
     "mean z disk size in y": table_zdisksize_y_mean,
     "mean z disk size in z": table_zdisksize_z_mean,
     "mean locs per z disk": table_locsperzdisk_mean,
     "mean x precision": table_xprec_mean,
     "mean y precision": table_yprec_mean,
     "mean z precision": table_zprec_mean,
     }  
)

output_loc = os.path.join(
        output_folder, f"table_k1.csv"
    )
df.to_csv(output_loc, index=False,encoding="utf-8-sig")
