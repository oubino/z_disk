# Overview

## Input data

You should have xyz localisation data for multiple FOVs.

Each FOV should be a .txt or .csv file.

Each localisation may also have additional features e.g. photon count

## Setup

In a new terminal with conda installed create new environment with required packages

```shell
conda create -n z_disk python=3.11 pyarrow scikit-learn matplotlib
conda activate z_disk
pip install open3d polars
```

Clone this directory to your files and move into the directory

```shell
git clone https://github.com/oubino/z_disk.git
cd z_disks
```

In experiments/ create a new folder for your experiment e.g. run_1/ 

Inside this folder, place all .txt or .csv data in a folder called data/

```bash
z_disks
├── README.md
├── experiments
│   ├── dummy
│       └── test.txt
│   └── run_1
│       └── data
│           ├── File_1.txt
│           └── File_2.txt
└── scripts
    ├── base.py
    ├── image_and_seg_to_pointcloud.py
    ├── pointcloud_to_image.py
    ├── separate_and_align.py
    └── visualise.py
```

## Running the pipeline

Make sure you have activated the correct environment before running scripts

1.  Convert each .txt/.csv file to an image

    ```shell
    python scripts/pointcloud_to_image.py [ARGS]
    ```
    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
        -x Name of the x column in the data
        -y Name of the y column in the data
        -z Name of the z column in the data
        -hx Size of histogram in x direction
        -hy Size of histogram in y direction
        -hz Size of histogram in z direction
        -s Delimeter separating the items - currently supported either comma (.csv) or tab (.txt)
        -bs Specify whether sizes above should be interpreted as the number of bins or the size of each bin, should be either bins or size
    ```

    The following args are optional
        -c Name of the channel column in the data 

2. Visualise the pointcloud data [Optional]

    ```shell
    python scripts/visualise.py [ARGS]
    ```

    The following arg is required:
        -i Path to the input parquet datastructure
        -d run DBSCAN with epsilon and min_samples, specificed in that order, separated by a space  

    Note on wsl2 for windows, had to use the following workaround

    ```shell
    export XDG_SESSION_TYPE=x11
    ```

3. Ilastik segmentation

    1. Create an empty folder called segmentations in output folder (this is where we will save the output from Ilastik)
    2. Open Ilastik and create a new project Other workflows > Pixel classification & Object classification (https://www.ilastik.org/documentation/pixelclassification/pixelclassification & https://www.ilastik.org/documentation/objects/objects)
    3. Load in the images from Output > Images
    4. If necessary to better visualise the images
        1. Right click the image and click edit properties
        2. Change normalize display to True 
        3. Keep range 0 to 1 and click ok
    5. Follow through the rest of the workflow
    6. Once have labelled the points in Object Classification then can export the labels 
        a. Right click on labels and click export labels
        b. Change the name to match the name of the file 
        c. Choose file type numpy
        d. Save in segmentations folder
        e. Repeat for each image

4. Combine each segmentation with the original .txt/.csv file to extract the localisations and return data in desired output format

    ```shell
    python scripts/image_and_seg_to_pointcloud.py
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
    ```

5. Separate the point-cloud into each object

    - In our case we have an optional alignment step as well using PCA

    ```shell
    python scripts/separate_and_align.py
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
    ```

    ```shell
    The following args are optional:
        -a If specified then aligns each z-disk with x axis - note that distances between points are not PERFECTLY preserved but errors are very small (errors in distances ~10^-11)
    ```

6. Visualise the .csv pointcloud data and optionally denoise (remove outliers) using DBSCAN

    - Note limitation of this is that have to manually go through the list of input files visualise each and decide [could instead script this]

    ```shell
    python scripts/visualise.py [ARGS]
    ```

    The following arg is required:
        -i Path to the input csv to be visualised
        -d run DBSCAN with epsilon and min_samples, specificed in that order, separated by a space 

7. Clean up after the visualising and denoising stage

    - Identifies files in segmented_z_disks/ but not in segmented_z_disks_denoised/ and copies across

    ```shell
    python scripts/clean_up_denoising.py
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
    ```

8. Filter out localisations with poor localisation precision

    ```shell
    python scripts/filter_loc_prec.py
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
        -p Filter out localisations with precision larger (i.e. worse) than this value
    ```

9. Visualise files to check

    - Note limitation of this is that have to manually go through the list of input files visualise each and decide [could instead script this]

    ```shell
    python scripts/visualise.py [ARGS]
    ```

    The following arg is required:
        -i Path to the input file to be visualised

10. Copy across checked files

    ```shell
    python scripts/remove_files_after_visualisation.py [ARGS]
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
        -i Input files to copy across
    ```

## Output

Output:
    - Folder of xyz data for each object as a .csv file optionally aligned along the x axis

The output directory will look as follows

```bash
...
├── output
│   ├── datastructures
│   │   └── File_1.parquet
│   ├── images
│   │   └── File_1.npy
│   ├── segmentations
│   │   └── File_1.npy
│   ├── segmented_pointclouds
|   │   └── File_1.csv
│   └── segmented_z_disks
|       ├── File_1_zdisk_0.csv
│       └── File_1_zdisk_1.csv
...
```

Each of the directories will be automatically generated apart from segmentations
This should be generated by Ilastik or another segmentation software and the files should have the same name as the original files containing a segmentation for the image i.e. a pixel level label

## Combine with PERPL

For PERPL respository see: https://github.com/AlistairCurd/PERPL-Python3/tree/master

1. Install perpl into z_disk environment

    ```shell
    pip install perpl
    ```

2. Extract relative positions

    ```shell
    python scripts/perpl_rel_posns.py
    ```

    ```shell
    The following args are required:
        -e Name of the experiment folder (e.g. dummy)
    ```

    ```shell
    The following args are optional:
        -f Filter distance
        -lf Filter by number of localisations
        -n Number of nearest neighbours
        -b Bin size for histogram plots
        -v Visualise the raw data AND the localisation precision
    ```

3. Modelling


    NOTE X and Y are flipped compared to work by A.C.
