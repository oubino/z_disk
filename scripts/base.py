"""Datastruc module.

This module contains definitions of the datastructure the
SMLM dataitem will be parsed as.

"""

import ast
import os

import numpy as np
import polars as pl
import pyarrow.parquet as pq

class item:
    """smlm datastructure.

    This is the basic datastructure which will contain all the information
    from a point set that is needed

    Attributes:
        name (string) : Contains the name of the item
        df (polars dataframe): Dataframe with the data contained, containing
            columns: 'channel' 'frame' 'x' 'y' 'z'.
            If manual annotation is done an additional column 'gt_label'
            will be present
        channels (list): list of ints, representing channels user wants
            to consider in the original data
        histo (dict): Dictionary of 2D or 3D arrays. Each key corresponds
            to the channel for which the histogram
            contains the relevant binned data, in form [X,Y,Z]
            i.e. histo[1] = histogram of channel 1 localisations.
            Note that if considering an image, need to transpose
            the histogram to follow image conventions.
        histo_edges (tuple of lists; each list contains floats):
            Tuple containing (x_edges,y_edges) or (x_edges, y_edges, z_edges)
            where x/y/z_edges are list of floats, each representing the
            edge of the bin in the original space.
            e.g. ([0,10,20],[13,25,20],[2,3,4])
        histo_mask (numpy array): Array containing integers where each should
            represent a different label of the MANUAL segmentation
            0 is reserved for background, is of form [X,Y,Z]
        bin_sizes (tuple of floats): Size of bins of the histogram
            e.g. (23.2, 34.5, 21.3)
    """

    def __init__(
        self,
        name,
        df,
        channels,
        histo={},
        histo_edges=None,
        histo_mask=None,
        bin_sizes=None,
    ):
        self.name = name
        self.df = df
        self.channels = channels
        self.histo = histo
        self.histo_edges = histo_edges
        self.histo_mask = histo_mask
        self.bin_sizes = bin_sizes

    def coord_2_histo(
        self,
        histo_size,
    ):
        """Converts localisations into histogram of desired size,
        with option to plot the image (histo.T).
        Note the interpolation is only applied for visualisation,
        not for the actual data in the histogram!

        Args:
            histo_size (tuple): Tuple representing number of
                bins/pixels in x,y,z
            cmap (list of strings) : The colourmaps used to
                plot the histograms
            vis_interpolation (string): How to inerpolate
                the image for visualisation"""

        # get max and min x/y/(z) values
        df_max = self.df.max()
        df_min = self.df.min()

        x_bins, y_bins, z_bins = histo_size
        x_max = df_max["x"][0]
        y_max = df_max["y"][0]
        x_min = df_min["x"][0]
        y_min = df_min["y"][0]
        z_max = df_max["z"][0]
        z_min = df_min["z"][0]

        # if instead want desired bin size e.g. 50nm, 50nm, 50nm
        # number of bins required for desired bin_size
        # note need to check if do this that agrees with np.digitize
        # and need to make sure that same issue we had before
        # with the last localisation is dealt with
        # x_bins = int((self.max['x'] - self.min['x']) / bin_size[0])
        # y_bins = int((self.max['y'] - self.min['y']) / bin_size[1])
        # z_bins = int((self.max['z'] - self.min['z']) / bin_size[2])

        # size of actual bins, given the number of bins (should be
        # very close to desired tests size)
        x_bin_size = (x_max - x_min) / x_bins
        y_bin_size = (y_max - y_min) / y_bins
        # need to increase bin size very marginally to include last localisation
        x_bin_size = x_bin_size * 1.001
        y_bin_size = y_bin_size * 1.001
        # location of edges of histogram, based on actual tests size
        x_edges = [x_min + x_bin_size * i for i in range(x_bins + 1)]
        y_edges = [y_min + y_bin_size * i for i in range(y_bins + 1)]
        # treat z separately, as often only in 2D
        z_bin_size = (z_max - z_min) / z_bins
        # need to increase bin size very marginally to include last localisation
        z_bin_size = z_bin_size * 1.001
        z_edges = [z_min + z_bin_size * i for i in range(z_bins + 1)]

        # size per tests in nm; location of histo edges in original space
        self.bin_sizes = (x_bin_size, y_bin_size, z_bin_size)
        self.histo_edges = (x_edges, y_edges, z_edges)

        print("-- Bin sizes -- ")
        print(self.bin_sizes)

        # 3D histogram for every channel, assigned to self.histo (dict)
        for chan in self.channels:
            df = self.df.filter(
                pl.col("channel") == chan
            )
            # This is dimensions D x N
            sample = np.array((df["x"], df["y"], df["z"]))
            # We need in dimensions N x D
            sample = np.swapaxes(sample, 0, 1)
            # (D, N) where D is self.dim and N is number of
            # localisations
            self.histo[chan], _ = np.histogramdd(sample, bins=self.histo_edges)

        # work out pixel for each localisations
        self._coord_2_pixel()


    def _coord_2_pixel(self):
        """Calculate the pixels corresponding to each localisation"""

        # drop pixel columns if already present
        for col in ["x_pixel", "y_pixel", "z_pixel"]:
            if col in self.df.columns:
                self.df = self.df.drop(col)

        # necessary for pd.eval below
        df_min = self.df.min()
        x_min = df_min["x"][0]
        y_min = df_min["y"][0]
        z_min = df_min["z"][0]
        x_pixel_width, y_pixel_width, z_pixel_width = self.bin_sizes

        # calculate pixel indices for localisations
        self.df = self.df.select(
            [
                pl.all(),
                pl.col("x").map(lambda q: (q - x_min) / x_pixel_width).alias("x_pixel"),
                pl.col("y").map(lambda q: (q - y_min) / y_pixel_width).alias("y_pixel"),
                pl.col("z").map(lambda q: (q - z_min) / z_pixel_width).alias("z_pixel"),
            ]
        )
        # floor the pixel locations
        self.df = self.df.with_columns(pl.col("x_pixel").cast(int, strict=True))
        self.df = self.df.with_columns(pl.col("y_pixel").cast(int, strict=True))
        self.df = self.df.with_columns(pl.col("z_pixel").cast(int, strict=True))


    def mask_pixel_2_coord(self, img_mask: np.ndarray) -> pl.DataFrame:
        """For a given mask over the image (value at each pixel
        normally representing a label), return the dataframe with a column
        giving the value for each localisation. Note that it is
        assumed that the img_mask is a mask of the image,
        therefore have to transpose img_mask for it to be in the same
        configuration as the histogram

        Note we also use this for  labels and when
        the img_mask represents probabilities.

        Args:
            img_mask (np.ndarray): Mask over the image -
                to reiterate, to convert this to histogram space need
                to transpose it

        Returns:
            df (polars dataframe): Original dataframe with
                additional column with the predicted label"""

        # list of mask dataframes, each mask dataframe contains
        # (x,y,label) columns
        # transpose the image mask to histogram space
        histo_mask = np.transpose(img_mask, (2, 1, 0))

        # create dataframe
        flatten_mask = np.ravel(histo_mask)
        mesh_grid = np.meshgrid(
            range(histo_mask.shape[0]), range(histo_mask.shape[1]), range(histo_mask.shape[2])
        )
        x_pixel = np.ravel(mesh_grid[1])
        y_pixel = np.ravel(mesh_grid[0])
        z_pixel = np.ravel(mesh_grid[2])
        label = flatten_mask
        data = {"x_pixel": x_pixel, "y_pixel": y_pixel, "z_pixel":z_pixel, "gt_label": label}
        mask_df = pl.DataFrame(
            data,
        ).sort(["x_pixel", "y_pixel", "z_pixel"])

        # join mask dataframe
        df = self.df.join(mask_df, how="inner", on=["x_pixel", "y_pixel", "z_pixel"])

        return df


    def save_df_to_csv(
        self, csv_loc, drop_zero_label=False,
    ):
        """Save the dataframe to a .csv with option to:
                drop positions which are background
                drop the column containing pixel information
                save additional column with labels for each
                    localisation

        Args:
            csv_loc (String): Save the csv to this location
            drop_zero_label (bool): If True then only non zero
                label positions are saved to csv
            save_chan_label (bool) : If True then save an
                additional column for each localisation
                containing the label for each channel

        Raises:
            NotImplementedError: This method is not implemented yet
            ValueError: If try to drop zero label when none is present"""

        save_df = self.df

        save_df = save_df.drop("x_pixel")
        save_df = save_df.drop("y_pixel")
        save_df = save_df.drop("z_pixel")

        # drop rows with zero label
        if drop_zero_label:
            if self.gt_label_scope == "loc":
                save_df = save_df.filter(pl.col("gt_label") != 0)
            else:
                raise ValueError("Can't drop zero label as no gt label column")

        # save to location
        save_df.write_csv(csv_loc)

    def save_to_parquet(
        self,
        save_folder,
        drop_zero_label=False,
        overwrite=False,
    ):
        """Save the dataframe to a parquet with option to drop positions which
           are background 

        Args:
            save_folder (String): Save the df to this folder
            drop_zero_label (bool): If True then only non zero
                label positions are saved to parquet
            overwrite (bool): Whether to overwrite

        Raises:
            ValueError: If try to drop zero label but no gt label; If the
                gt label and gt label scope are incompatible; If try
                to overwrite without manually saying want to do this
        """

        save_df = self.df

        # drop rows with zero label
        if drop_zero_label:
            save_df = save_df.filter(pl.col("gt_label") != 0)

        # convert to arrow + add in metadata if doesn't exist
        arrow_table = save_df.to_arrow()

        # convert gt label map to bytes
        old_metadata = arrow_table.schema.metadata

        meta_data = {
            "name": self.name,
            "channels": str(self.channels),
            "bin_sizes": str(self.bin_sizes),
        }

        # add in label mapping
        # change
        # merge existing with new meta data
        merged_metadata = {**meta_data, **(old_metadata or {})}
        arrow_table = arrow_table.replace_schema_metadata(merged_metadata)
        save_loc = os.path.join(
            save_folder, self.name + ".parquet"
        )  # note if change this need to adjust annotate.py
        if os.path.exists(save_loc) and not overwrite:
            raise ValueError(
                "Cannot overwite. If you want to overwrite please set overwrite==True"
            )
        pq.write_table(arrow_table, save_loc)

        # To access metadata write
        # parquet_table = pq.read_table(file_path)
        # parquet_table.schema.metadata ==> metadata
        # note if accessing keys need
        # parquet_table.schema.metadata[b'key_name'])
        # note that b is bytes

    def load_from_parquet(self, input_file):
        """Loads item saved as .parquet file

        Args:
            input_file (string) : Location of the .parquet file to
                load dataitem from"""

        # read in parquet file
        arrow_table = pq.read_table(input_file)

        # print("loaded metadata", arrow_table.schema.metadata)

        # metadata
        name = arrow_table.schema.metadata[b"name"].decode("utf-8")

        # convert string keys to int keys for the mapping
    
        channels = arrow_table.schema.metadata[b"channels"]
        channels = ast.literal_eval(channels.decode("utf-8"))
        bin_sizes = arrow_table.schema.metadata[b"bin_sizes"]
        bin_sizes = ast.literal_eval(bin_sizes.decode("utf-8"))
        df = pl.from_arrow(arrow_table)

        self.__init__(
            name=name,
            df=df,
            channels=channels,
            bin_sizes=bin_sizes,
        )

    def render_histo(self):
        """Render the histogram from the .parquet file

        Assumes localisations have associated x_pixel, y_pixel & z_pixel already.


        Returns:
            histo (np.histogram) : Histogram of the localisation data
        """

        histos = []

        df_max = self.df.max()
        x_bins = df_max["x_pixel"][0] + 1
        y_bins = df_max["y_pixel"][0] + 1
        z_bins = df_max["z_pixel"][0] + 1

        for chan in self.channels:
            df = self.df.filter(pl.col("channel") == chan)

            histo = np.zeros((x_bins, y_bins, z_bins))
            df = df.group_by(by=["x_pixel", "y_pixel", "z_pixel"]).count()
            x_pixels = df["x_pixel"].to_numpy()
            y_pixels = df["y_pixel"].to_numpy()
            z_pixels = df["z_pixel"].to_numpy()
            counts = df["count"].to_numpy()
            histo[x_pixels, y_pixels, z_pixels] = counts

            histos.append(histo)

        histo = np.stack(histos)

        return histo
