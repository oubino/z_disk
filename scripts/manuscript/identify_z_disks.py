import os
import polars as pl
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

dpi = 600
plt.rcParams["figure.dpi"] = dpi


for protein in ["ACTN2", "Z1Z2", "ZASP6"]:

    # print(f"----- Protein: {protein} ------ ")

    input_folder = os.path.join(
        f"experiments/{protein}/output/segmentations"
    )

    files = os.listdir(input_folder)

    files = np.sort(files)

    for file in files:

        file_path = os.path.join(input_folder, file)

        file = file.rstrip(".npy")

        if protein == "ACTN2":

            if file != "25_1":
                continue
            else:
               array = np.load(file_path)
               print(array.shape)
               plt.imshow(array[0,:,:,0])
               plt.show()
        
        elif protein == "Z1Z2":

            if file != "23_3":
                continue
            else:
               array = np.load(file_path)
               print(array.shape)
               plt.imshow(array[0,:,:,0])
               plt.show()
        
        elif protein == "ZASP6":

            if file != "24_4":
                continue
            else:
               array = np.load(file_path)
               print(array.shape)
               plt.imshow(array[0,:,:,0])
               plt.show()

     

raise ValueError("stop")

for protein in ["ACTN2", "Z1Z2", "ZASP6"]:

    # print(f"----- Protein: {protein} ------ ")

    input_folder = os.path.join(
        f"experiments/{protein}/output/segmented_pointclouds"
    )

    files = os.listdir(input_folder)

    files = np.sort(files)

    for file in files:

        file_path = os.path.join(input_folder, file)

        file = file.rstrip(".csv")

        coms = []
        fig, ax = plt.subplots()#= plt.figure()
        #ax = fig.add_subplot()
        ax.set_aspect("equal")
        ax.set_facecolor((0.0, 0.0, 0.0))
        ax.grid(False)
        size = 0.2

        df = pl.read_csv(file_path)

        if protein == "ACTN2":

            if file != "25_1":
                continue
            else:
               df1 = df.filter(pl.col("gt_label") == 1)
               df2 = df.filter(pl.col("gt_label") != 1)
        
        elif protein == "Z1Z2":

            if file != "23_3":
                continue
            else:
               df1 = df.filter(pl.col("gt_label") == 1)
               df2 = df.filter(pl.col("gt_label") != 1)
        
        elif protein == "ZASP6":

            if file != "24_4":
                continue
            else:
               df1 = df.filter(pl.col("gt_label") == 3)
               df2 = df.filter(pl.col("gt_label") != 3)

        array = np.array([df1["x"], df1["y"] , df1["z"]])
        ax.scatter(array[0,:], array[1,:], s=size, linewidths=size, c="r")

        array = np.array([df2["x"], df2["y"] , df2["z"]])
        ax.scatter(array[0,:], array[1,:], s=size, linewidths=size, c="w")

        file = file.rstrip(".csv")
        plt.savefig(f"manuscript_figures/identify_z_disk_in_A/{protein}_{file}.png")

