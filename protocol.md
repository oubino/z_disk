# Protocol (26/11/25)

This is the protocol used to generate the results for the iPALM z-disk data.

Replace PROTEIN with ACTN2, Z1Z2 or ZASP6 in protocol below.
Replace FILENAME accordingly below as well.

1. Make empty folder in experiments/ called PROTEIN

2. Make empty folder in experiments/PROTEIN/ called data

3. Copy in data from Janelia/Oli/PROTEIN/renamed to experiments/PROTEIN/data

4. Convert to images
    ```bash
    python scripts/pointcloud_to_image.py -e PROTEIN -x "Group X Position" -y "Group Y Position" -z "Group Z Position" -hx 50 -hy 50 -hz 50 -s tab -bs size
    ```

5. Visualise each one – replace FILENAME with 16_2, 17_2, etc..
    ```bash
    python scripts/visualise.py -i experiments/PROTEIN/output/datastructures/FILENAME.parquet
    ```

6. Create an empty folder called segmentations in output folder (this is where we will save the output from Ilastik)

7. Run Ilastik (modify path to run_ilastik.sh accordingly, or simply open up the GUI if in windows, mac)
    ```bash
    bash ../ilastik-1.4.0-Linux/run_ilastik.sh
    ```

8. In Ilastik
    1. create a new project Other workflows > Pixel classification & Object classification (https://www.ilastik.org/documentation/pixelclassification/pixelclassification & https://www.ilastik.org/documentation/objects/objects) [Ilastik_video_1]
    2. Load in the images from Output > Images [Ilastik_video_2]
    3. If necessary to better visualise the images [Ilastik_video_3]
                    1. Right click the image and click edit properties
                    2. Change normalize display to True 
                    3. Keep range 0 to 1 and click ok
            d. Feature selection [Ilastik_video_4]
    4. Training [Ilastik_video_5]
    5. Thresholding [Ilastik_video_6]
    6. Object feature selection [Ilastik_video_7]
    7. Object classification & export labels [Ilastik_video_8 – note that wrong output folder is shown in this video should have been experiments/ACTN2/output/segmentations]
        1. Right click on labels and click export labels
        2. Change the name to match the name of the file 
        3. Choose file type numpy
        4. Save in segmentations folder
        5. Repeat for each image
    
9. Convert back to point-clouds
    ```bash
	python scripts/image_and_seg_to_pointcloud.py -e PROTEIN
    ```

10. Separate and align
    ```bash
	python scripts/separate_and_align.py -e PROTEIN -a
    ```

11. Visualise and denoise the z disks
    1. Manually go through each z disk file name and visualise each one…[could script so does automatically but for the moment leaving like this]
        ```bash
        python scripts/visualise.py -i experiments/PROTEIN/output/segmented_z_disks/FILENAME.csv
        ```

    2. If the z disk requires denoising, run visualisation again with DBSCAN
        ```bash
        python scripts/visualise.py -i experiments/PROTEIN/output/segmented_z_disks/FILENAME.csv -d 120 3
        ```

    3. Repeat until identify parameters [range: 50-150 nm, 3-7 minpts] that extract the z-disk, at this point click [Y] to everything

    4. Double check worked by comparing
        ```bash
        python scripts/visualise.py -I experiments/PROTEIN/output/segmented_z_disks/FILENAME.csv -d 50 3
        python scripts/visualise.py -i experiments/PROTEIN/output/segmented_z_disks_denoised/FILENAME.csv
        ```

12. Double check by visualising each z disk in denoised
    ```bash
    python scripts/visualise.py -I experiments/PROTEIN/output/segmented_z_disks_denoised/FILENAME.csv
    ```

13. Clean up denoising
    ```bash
	python scripts/clean_up_denoising.py -e PROTEIN
    ```

14. Filter data
    ```bash
	python scripts/filter_loc_prec.py -e PROTEIN -p 5
    python scripts/filter_loc_prec.py -e PROTEIN -p 6
    python scripts/filter_loc_prec.py -e PROTEIN -p 7
    python scripts/filter_loc_prec.py -e PROTEIN -p 8
    python scripts/filter_loc_prec.py -e PROTEIN -p 9
    python scripts/filter_loc_prec.py -e PROTEIN -p 10
    python scripts/filter_loc_prec.py -e PROTEIN -p 15
    ```

15. Visualise each filtered dataitem to decide whether to include based on 15.0nm filter
    ```bash
    python scripts/visualise.py -i experiments/PROTEIN/output/segmented_z_disks_denoised_filtered/15.0nm_filter/FILENAME.csv
    ```

16. Copy across checked files…
    ```bash
    python scripts/remove_files_after_visualisation.py -e PROTEIN -i all 
    ```

17. Calculate relative positions, with varying num localisations filter
    ```bash
    python scripts/perpl_rel_posns.py -e PROTEIN -f 150 -lf 150
    python scripts/perpl_rel_posns.py -e PROTEIN -f 150 -lf 100
    python scripts/perpl_rel_posns.py -e PROTEIN -f 150 -lf 10
    python scripts/perpl_rel_posns.py -e PROTEIN -f 150 -lf 5
    python scripts/perpl_rel_posns.py -e PROTEIN -f 150 -lf 1
    ```

19. In experiment/ folder (i.e. one directory above output folder) create a folder called perpl_config/ which contains the config.yaml file

    i.e.

    ```bash
    ...
    ├── data
    ├── output
    └── perpl_config
       └── config.yaml 
    ...
    ```

20. Perform an an exhaustive sweep of all possible models within a set range. Run

    ```shell
    python scripts/gen_sweep_configs.py -e PROTEIN
    ```

    to generate all the possible model config files.

21. Then to model the data run

    ```shell
    python scripts/perpl_modelling.py -e PROTEIN
    ```

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

22. Generate manuscript figures and tables
    ```bash
    python scripts/manuscript/script_K1.py
    python scripts/manuscript/script_A1.py
    python scripts/manuscript/best_models_table.py
    ```
