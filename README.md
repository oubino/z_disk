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
    python scripts/pointcloud_to_image.py
    ```

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

    The following args are optional:

        -c Name of the channel column in the data 

2. Visualise the pointcloud data [Optional]

    ```shell
    python scripts/visualise.py
    ```

    The following arg is required:

        -i Path to the input parquet datastructure
    
    The following args is optional:

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
    
    We used Ilastik to generate the segmentations.
    They could be generated using another segmentation software.
    Regardless, the segmentation file for each image (containing pixel level labels) should have the same name as the original image file and should be a .npy file.

4. Combine each segmentation with the original .txt/.csv file to extract the localisations and return data in desired output format

    ```shell
    python scripts/image_and_seg_to_pointcloud.py
    ```

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)


5. Separate the point-cloud into each object

    ```shell
    python scripts/separate_and_align.py
    ```

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)

    The following args are optional:

        -a If specified then aligns each z-disk with x axis
        
    Alignment is via PCA. 
    Also note that distances between points are not PERFECTLY preserved during alignment.
    However, errors are very small (errors in distances ~10^-11)

6. Visualise the .csv pointcloud data and optionally denoise (remove outliers) using DBSCAN

    Note limitation of this is that have to manually go through the list of input files visualise each and decide [could instead script this]

    ```shell
    python scripts/visualise.py
    ```

    The following arg is required:

        -i Path to the input csv to be visualised

    The following args are optional:

        -d run DBSCAN with epsilon and min_samples, specificed in that order, separated by a space 

7. Clean up after the visualising and denoising stage

    Identifies files in segmented_z_disks, that are not in segmented_z_disks_denoised and copies them across

    ```shell
    python scripts/clean_up_denoising.py
    ```

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)


8. Filter out localisations with poor localisation precision

    ```shell
    python scripts/filter_loc_prec.py
    ```

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)
        -p Filter out localisations with precision larger (i.e. worse) than this value


9. Visualise files to check

    Note limitation of this is that have to manually go through the list of input files visualise each and decide [could instead script this]

    ```shell
    python scripts/visualise.py
    ```

    The following arg is required:

        -i Path to the input file to be visualised

10. Copy across checked files

    ```shell
    python scripts/remove_files_after_visualisation.py
    ```

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)
        -i Input files to copy across. If just give all as the argument, copies across all files


## Output

Output:

    Folder of xyz data for each object as a .csv file optionally aligned along the x axis

The output directory should look as follows

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
│   ├── segmented_z_disks
|   │   ├── File_1_zdisk_0.csv
│   │   └── File_1_zdisk_1.csv
│   ├── segmented_z_disks_denoised
|   │   ├── File_1_zdisk_0.csv
│   │   └── File_1_zdisk_1.csv
│   ├── segmented_z_disks_denoised_filtered
|   │   ├── File_1_zdisk_0.csv
│   │   └── File_1_zdisk_1.csv
│   └── segmented_z_disks_denoised_filtered_vischecked
|       ├── File_1_zdisk_0.csv
|       └── File_1_zdisk_1.csv
|   
...
```

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

    The following args are required:

        -e Name of the experiment folder (e.g. dummy)

    The following args are optional:
    
        -f Filter distance
        -lf Filter by number of localisations
        -n Number of nearest neighbours
        -b Bin size for histogram plots
        -v Visualise the raw data AND the localisation precision


3. Modelling

    Note:  (i.e. X and Y are flipped compared to work by A.C.)
    
   1. In experiment/ folder (i.e. one directory above output folder) need a folder called perpl_config/ which contains the configuration files

        i.e.

        ```bash
        ...
        ├── data
        ├── output
        └── perpl_config
           ├── config.yaml 
           └── models
              ├── axial
               |   ├── model_0.yaml
               |   └── model_1.yaml
              └── transverse
                    ├── model_0.yaml
                    └── model_1.yaml
        ...
        ```

        For an example of a perpl_config folder see examples/ 

        The config.yaml file specifies generic information applicable to all the models being fit.

        Each model.yaml file specifies a specific model being fit to the data.

    2. For an exhaustive sweep of all possible models within a set range. Define the possible values for each parameter in the config.yaml file (see examples/config.yaml) and run 

        ```shell
        python scripts/gen_sweep_configs.py
        ```

        with the following required args

            -e Name of the experiment folder (e.g. dummy)

        to generate all the possible model config files.

    3. Then to model the data run
    
        ```shell
        python scripts/perpl_modelling.py
        ```

        This takes the following args:

            -e Name of the experiment folder (e.g. dummy) [required]
            -fh [optional: fit the histograms as well]

        This will generate output in this format

        ```bash
        ...
        ├── data
        ├── output
            └──perpl_modelling
               ├── axial
            |   ├── histograms/
            |   ├── kdes/
            |   ├── results_histograms.csv
            |   └── results_kdes.csv
               └── transverse
            |   ├── histograms/
            |   ├── kdes/
            |   ├── results_histograms.csv
            |   └── results_kdes.csv
        ...
        ```
    
    4. You can follow this workflow to model your data ![workflow](/resources/workflow_v0.1.svg).
