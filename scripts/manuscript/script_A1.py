# Generate figure k1 and table k1

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.patches as patches
import polars as pl
import pyarrow.parquet as pq
import matplotlib.animation as animation

plot_3d = False
plot_2d = True
planes = ["xy", "yz"]  # xy, yz
filter = False
generate_anim = False
output_folder = "manuscript_figures/figure_A1/"

for protein, file, file_zdisk in zip(["ACTN2", "Z1Z2", "ZASP6"], ["25_1", "23_3", "24_4"], ["_zdisk_1_aligned.csv", "_zdisk_1_aligned.csv", "_zdisk_3_aligned.csv"]):

    print(protein," ", file)

    file_zdisk = file + file_zdisk
    folder = f"experiments/{protein}/output/images/"


    # ------- PLOT OVERVIEW ------

    # plot with scale bar
    plt.figure(figsize=(20, 16))

    ax = plt.gca()

    # load in file and max project in z
    f = np.load(os.path.join(folder, file + ".npy"))
    # f[z, y, x, c]
    f = np.max(f, axis=0)  # max projection in z
    f = f[:, :, 0]    

    # load in scale bar
    df = pq.read_table(f"experiments/{protein}/output/datastructures/{file}" + ".parquet")
    bin_sizes = df.schema.metadata[b"bin_sizes"]
    bin_sizes = ast.literal_eval(bin_sizes.decode("utf-8"))
    x_bin_size, y_bin_size, z_bin_size = bin_sizes

    assert int(x_bin_size) == int(y_bin_size)

    #print("X and Y bin size in nm: ", int(x_bin_size))
    #print("2 μm : ", int(2000/x_bin_size), " pixels")

    scale_bar_x = int(2000 / x_bin_size)
    rect = patches.Rectangle((10, 10), width=scale_bar_x, height=20, facecolor="w")
    # Add the patch to the Axes
    ax.add_patch(rect)

    # auto-crop
    # non_empty_columns = np.where(f.max(axis=0)>0)[0]
    # non_empty_rows = np.where(f.max(axis=1)>0)[0]
    # cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    # f = f[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]

    # normalise the data
    norm = mpc.Normalize(vmin=0, vmax=1)

    #plt.text(21, 390, "2 μm", c="w", size=30)
    plt.axis("off")
    plt.imshow(f, cmap="gray", norm=norm, origin="upper")
    plt.savefig(
        os.path.join(output_folder, f"{protein}_overview.svg"),
        bbox_inches="tight",
        transparent=True,
        dpi=600,
    )
    plt.close()

    # ------- PLOT Z DISK ------

    folder = f"experiments/{protein}/output/segmented_z_disks/"
    f = pl.read_csv(os.path.join(folder, file_zdisk))

    if filter is not False:
        f = f.filter(
            pl.col("Group Sigma X Pos").str.strip_chars_start(" ").cast(pl.Float64)
            < filter,
            pl.col("Group Sigma Y Pos").str.strip_chars_start(" ").cast(pl.Float64)
            < filter,
            pl.col("Group Sigma Z").str.strip_chars_start(" ").cast(pl.Float64) < filter,
        )

    x, y, z = f["x", "y", "z"]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # ---- plot Z disk in 2D -----
    if plot_2d:
        for plane in planes:
            if plane == "xy":
                figsize = (6, 1)
            elif plane == "yz":
                figsize = (2.4, 4.8)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot()
            ax.set_aspect("equal")
            ax.set_facecolor((0.0, 0.0, 0.0))

            # Hide grid lines
            ax.grid(False)

            # Hide axes ticks
            #ax.set_xticks([])
            #ax.set_yticks([])

            size = 0.2
            if plane == "xy":
                ax.scatter(x, y, s=size, linewidths=size, c="w")
            elif plane == "yz":
                ax.scatter(y, z, s=size, linewidths=size,  c="w")

            # Create a Rectangle patch
            #if plane == "xy":
            #    rect = patches.Rectangle((-180, 18130), width=100, height=20, facecolor="w")
            #    plt.text(-190, 18100, "100 nm", c="w", size=7)
            #    plt.xlim(-220, 1500)
            #elif plane == "yz":
            #    rect = patches.Rectangle((18110, 0), width=50, height=10, facecolor="w")
            #    plt.text(18108, -15, "50 nm", c="w", size=10)

            # Add the patch to the Axes
            #ax.add_patch(rect)

            print("X range: ", np.max(x) - np.min(x))
            print("Y range: ", np.max(y) - np.min(y))
            print("Z range: ", np.max(z) - np.min(z))

            fig.set_dpi(1200)

            plt.savefig(
                os.path.join(output_folder, protein + file_zdisk.rstrip(".csv") + plane + ".svg"),
                bbox_inches="tight",
                dpi=1200,
            )

    # ---- plot Z disk in 3D -----
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # ax.set_aspect("equal")
        ax.set_facecolor((0.0, 0.0, 0.0))
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Remove the grey panes (make them transparent)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Remove all 12 lines of the 3D bounding box
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_visible(False)
            axis.set_pane_color((1, 1, 1, 0))

        ax.set_box_aspect(
            (np.ptp(x), np.ptp(y), np.ptp(z))
        )  # aspect ratio is 1:1:1 in data space
        scatter = ax.scatter(x, y, z, s=5, c="w")

        # plt.show()
        option_1 = True
        option_2 = False
        option_3 = False
        option_4 = False
        option_5 = False

        if generate_anim:

            ## Set initial view
            ax.view_init(elev=20, azim=140)

            # --- Animation function ---
            def update(frame):

                if option_1:
                    elev = azim = roll = frame

                if option_2:
                    rad = np.deg2rad(frame)
                    azim = frame
                    elev = 30 * np.sin(rad) + 30
                    roll = 15 * np.cos(2 * rad)

                if option_3:
                    rad = np.deg2rad(frame)
                    azim = frame
                    elev = 45 * np.cos(rad)
                    roll = (frame * 0.5) % 360

                if option_4:
                    rad = np.deg2rad(frame)
                    azim = frame
                    elev = 25 * np.sin(rad) + 12 * np.sin(3 * rad) + 30
                    roll = 20 * np.sin(1.5 * rad) + 5 * np.cos(4 * rad)

                if option_5:
                    rad = np.deg2rad(frame)
                    # unit circle motion
                    u = np.array(
                        [np.cos(rad), np.sin(rad), 0.2 * np.sin(2 * rad)]
                    )  # add z wobble
                    # derive angles (approx): azimuth = atan2(y,x), elevation = atan2(z, sqrt(x^2+y^2))
                    azim = np.rad2deg(np.arctan2(u[1], u[0])) % 360
                    elev = np.rad2deg(np.arctan2(u[2], np.hypot(u[0], u[1])))
                    roll = (frame * 0.3) % 360

                ax.view_init(elev=elev, azim=azim, roll=roll)
                return (scatter,)

            # --- Create animation ---
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=360,  # rotate 360 degrees
                interval=20,  # milliseconds between frames
                blit=True,
            )

            # --- Save movie (requires ffmpeg installed) ---
            output_movie = os.path.join(output_folder, f"{protein}_z_disk.gif")
            anim.save(output_movie, fps=30, dpi=150)
