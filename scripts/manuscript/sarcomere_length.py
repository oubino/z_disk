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
        f"experiments/{protein}/output/segmented_pointclouds"
    )

    files = os.listdir(input_folder)

    files = np.sort(files)

    for file in files:

        file_path = os.path.join(input_folder, file)

        file.rstrip(".csv")

        coms = []
        fig, ax = plt.subplots()#= plt.figure()
        #ax = fig.add_subplot()
        ax.set_aspect("equal")
        ax.set_facecolor((0.0, 0.0, 0.0))
        ax.grid(False)
        size = 0.2

        df = pl.read_csv(file_path)

        array = np.array([df["x"], df["y"] , df["z"]])

        ax.scatter(array[0,:], array[1,:], s=size, linewidths=size, c="w")

        coms = df.group_by(pl.col("gt_label")).mean()["x","y","z"].to_numpy()

        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coms)
        distances, indices = nbrs.kneighbors(coms)

        ax.scatter(coms[:,0], coms[:,1], s=2, c="r", marker="x")

        # -------------

        text_x = []

        for i, (dist, index) in enumerate(zip(distances,indices)):

            src = coms[i]

            # ignore itself
            dist = dist[1:]
            index = index[1:]

            #print("dist: ", dist)

            for j in index:

                dest = coms[j]


                x0 = src[0]
                y0 = src[1]
                x1 = dest[0]
                y1 = dest[1]

            
                dist = (y1 - y0)**2 + (x1 - x0)**2
                dist = dist ** 0.5
                dist = np.abs(np.round(dist, 1))

                #print("dist: ", dist)

                xtext = (x0+x1)/2
                ytext = (y0+y1)/2

                if xtext not in text_x:
                #    ax.text(xtext, ytext, dist, c="r", size= 5)
                #    ax.plot([x0,x1], [y0,y1], c="r", linewidth=.5)
                    text_x.append(xtext)


        # ------------
        
        print(protein, " ", file)
        
        figsize = fig.get_size_inches()
        figsize *= dpi

        print(figsize)

        point1 = ax.transData.inverted().transform((100,100))
        point2 = ax.transData.inverted().transform((100,200))
        diff1 = point2 - point1

        point1 = ax.transData.inverted().transform((100,100))
        point2 = ax.transData.inverted().transform((200,100))
        diff2 = point2 - point1

        diff = max(diff1[1], diff2[0])

        print("1 pixels: ", diff/100, "nm")

        file = file.rstrip(".csv")
        plt.savefig(f"manuscript_figures/sarcomere_length/{protein}_{file}.png")

