# Generate figure k1 and table k1

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import polars as pl
import pyarrow.parquet as pq
import matplotlib.animation as animation

protein = "ACTN2"
file = "25_1"

# ------- PLOT OVERVIEW ------

output_folder = "manuscript_figures/figure_A1/"
folder = f"experiments/{protein}/output/images/"

# load in file and max project in z
f = np.load(os.path.join(folder, file + ".npy"))
# f[z, y, x, c]
f = np.max(f, axis=0) # max projection in z
f = f[:, :, 0]

# manual crop
f = f[125:550, 25:375] 

# load in scale bar
df = pq.read_table(f"experiments/{protein}/output/datastructures/{file}" + ".parquet")
bin_sizes = df.schema.metadata[b"bin_sizes"]
bin_sizes = ast.literal_eval(bin_sizes.decode("utf-8"))
x_bin_size, y_bin_size, z_bin_size = bin_sizes

assert(int(x_bin_size) == int(y_bin_size))

scale_bar_x = int(2000/x_bin_size)
f[395:405, 20:20+scale_bar_x] = 1.0

# auto-crop
#non_empty_columns = np.where(f.max(axis=0)>0)[0]
#non_empty_rows = np.where(f.max(axis=1)>0)[0]
#cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
#f = f[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]

# normalise the data
norm = mpc.Normalize(vmin=0, vmax=1)

# plot with scale bar
plt.figure(figsize=(20, 16))   
plt.text(21, 390, "2 μm", c='w', size= 30)
plt.axis('off')
plt.imshow(f, cmap="gray", norm=norm, origin="upper")
plt.savefig(os.path.join(output_folder, "actn_overview.png"))
plt.close()

# ------- PLOT Z DISK ------

folder = f"experiments/{protein}/output/segmented_z_disks/"
file = file + "_zdisk_1_aligned.csv"

f = pl.read_csv(os.path.join(folder, file))

plot_3d = False
plot_2d = True
plane = "xy" # xy, yz
filter = False

if filter is not False:
    f = f.filter(
        pl.col("Group Sigma X Pos").str.strip_chars_start(" ").cast(pl.Float64) < filter,
        pl.col("Group Sigma Y Pos").str.strip_chars_start(" ").cast(pl.Float64) < filter,
        pl.col("Group Sigma Z").str.strip_chars_start(" ").cast(pl.Float64) < filter,
    )

x, y, z = f["x","y","z"]
x = np.array(x)
y = np.array(y)
z = np.array(z)

if plot_2d:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_facecolor((0.0, 0.0, 0.0))
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if plane == "xy":
        ax.scatter(x, y, s =5, c="w")
    elif plane == "yz":
        ax.scatter(y, z, s =5, c="w") 
    plt.show()

if plot_3d:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.set_aspect("equal")
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

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # aspect ratio is 1:1:1 in data space
    scatter = ax.scatter(x, y, z, s =5, c="w")

    #plt.show()

    ## Set initial view
    ax.view_init(elev=20, azim=140)

    # --- Animation function ---
    def update(frame):
        ax.view_init(elev=frame, azim=frame, roll=frame)
        return scatter,

    # --- Create animation ---
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=360,       # rotate 360 degrees
        interval=20,      # milliseconds between frames
        blit=True
    )

    ### 

    # --- Save movie (requires ffmpeg installed) ---
    output_movie = os.path.join(output_folder, "actn_z_disk.gif")
    anim.save(output_movie, fps=30, dpi=150)

    print(np.max(x) - np.min(x))
    print(np.max(y) - np.min(y))
    print(np.max(z) - np.min(z))

    #plt.show()