from __future__ import annotations
import os
from datetime import datetime
import warnings
import itertools
import csv
import numpy as np
from functools import partial
from typing import List, Tuple
from skimage import img_as_ubyte, img_as_float64
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rotate, resize, downscale_local_mean
from skimage.draw import line, disk
from skimage.morphology import binary_erosion, binary_dilation, ellipse, area_closing
from skimage.filters import threshold_otsu
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops
from scipy.signal import fftconvolve, find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from .hexagon_math import HexagonMath
from .options import ImageType, ChannelNames, OrganoidProperties, SCALE_RATIO, PATH_DELIMITER, VERSION
import utils.usefull_functions as uf
from utils.loggin_utils import log_method
import logging
logger = logging.getLogger(__name__)

class Exp:

    substrates_refs: list[Substrate] #enable autocompletion and will hold Substrate instances
    
    @log_method()
    def __init__(self, name: str, path: str, days: List[str], substrates: List[str], filename_pattern: str, params: List[dict], start_date: str=None, protocol_path: str=None):
        """Initialize an experiment.

        An Exp instance represents the top-level container of the image
        processing pipeline. It creates and manages Substrate objects,
        which themselves create Day objects.

        Parameters
        ----------
        name : str
            Name of the experiment.

        path : str
            Root directory containing all experimental data.

        days : list of str
            List of day identifiers (e.g. ["D1", "D2"]).

        substrates : list of str
            List of substrate/condition names.

        filename_pattern : str
            Pattern used to identify image files.

        params : list of dict
            List of acquisition parameters (one per substrate and day).

        start_date : str, optional
            Experiment start date.

        protocol_path : str, optional
            Path to protocol documentation.

        Notes
        -----
        - Automatically creates Substrate instances.
        - Substrate instances create Day instances.
        - Day instances create Hive instances.
        """
        
        logger.info(f"{name} | Creating Exp instance")

        self.name = name
        self.path = path
        self.filename_pattern = filename_pattern
        self.start_date = start_date
        self.protocol_path = protocol_path
        self.substrates_names = substrates.copy()
        self.days_names = days
        self.filenames = [[None for _ in range(len(days))] for _ in range(len(substrates))]
        self.params = params
        self.substrates_refs = []

        #Verify path existence for experiment & substrate
        if os.path.isdir(self.path) == False:
            logger.error(f"{self.name} | Experiment folder not found: {path}")
            raise ValueError(f"{self.name} | Experiment folder not found: {path}")
        else:
            for s in substrates:
                s_path = self.path + PATH_DELIMITER + s
                if os.path.isdir(s_path) == False:
                    logger.warning(f"{self.name} | Substrate folder not found: substrate={s}, path={s_path}. --> Removed from instance.\n\tVerify susbtrate name in folder and input csv file: they should be identical")
                    self.substrates_names.remove(s)
        
        #Create hexagon instance for each Microscope/Magnification/ImageDim encoutered
        self.hexagon = self.create_hexagon_masks()   
        
        #Create and store daughter class
        for_child = {"exp_path": path, "exp_name": name, "pattern": filename_pattern}
        for s in substrates:
            substrate = Substrate(s, for_child, self.hexagon, days, params)
            self.substrates_refs.append(substrate)

    def create_hexagon_masks(self):
        """Generate and store all hexagonal masks required for hive detection.

        This method computes geometric hexagon masks based on the unique
        acquisition parameter combinations defined in `self.params`.
        For each unique combination of:

            (Design, Microscope, Magnification, Dimension)

        two masks are generated:
            1. Full-scale hexagon mask
            2. Rescaled hexagon mask (used for downscaled processing)

        The masks are stored in a dictionary indexed by the acquisition
        parameter tuple and saved as an attribute of the Exp instance:

            self.hexagon[(Design, Microscope, Magnification, Dimension)] = {
                "hexagon_full_scale": ndarray,
                "hexagon_rescale": ndarray
            }

        Returns
        -------
        dict
            Dictionary containing all computed hexagon masks.

        Notes
        -----
        - Geometry depends on acquisition metadata (design, microscope,
        magnification, image dimensions).
        - Rescaled masks are computed using the global `SCALE_RATIO`.
        - These masks are accessed at Day and Hive levels through
        parent references (self.parent / self.parent.parent).
        - Must be called before cropping or rotation-based alignment.
        """

        hexagon_mask = {}
        
        hexagon_params = uf.unique_list([(elt["Design"],elt["Microscope"],elt["Magnification"],elt["Dimension"]) for elt in self.params])
        
        for elt in hexagon_params:

            if "Plate" in elt:
                continue

            img_shape = [ int(i) for i in elt[3].split("x") ]
            center = uf.get_img_center(img_shape)
            hive_height, n_expect, um2pxl = uf.assign_dim(elt[0], elt[1], elt[2], elt[3])
            side = HexagonMath.get_side_from_height(hive_height)*um2pxl
            img_shape_rs = [ round(i/SCALE_RATIO) for i in img_shape]
            center_rs = uf.get_img_center(img_shape_rs)
            side_rs = HexagonMath.get_side_from_height(hive_height/SCALE_RATIO)*um2pxl
            hexagon_mask[elt] = {
                "hexagon_full_scale": HexagonMath.hexagon_outline_ndarray(img_shape, center, side, thickness=5, return_type='hexagon'),
                "hexagon_rescale": HexagonMath.hexagon_outline_ndarray(img_shape_rs, center_rs, side_rs, thickness=5, return_type='hexagon'),
                "side_pxl": HexagonMath.get_side_from_height(hive_height)*um2pxl,
                "side_um": side,
                "height_pxl": hive_height*um2pxl,
                "height_um": hive_height,
                "diagonal_pxl": HexagonMath.get_diagonal_from_height(hive_height)*um2pxl,
                "diagonal_um": HexagonMath.get_diagonal_from_height(hive_height)
                }
        
        return hexagon_mask

    def __repr__(self):
        """Defines how to print the Exp class: Exp(ExpName, Path, [Substrate1, ..., SubstrateN], [D1, ..., DN], StartingDate, Protocol)"""
        
        days = ",".join(self.days_names)
        substrates = ",".join(self.substrates_names)
        rep = "Exp(" + self.name + ", " + self.path + " , [" + substrates + "], [" + days + "], " + str(self.start_date) + ", " + str(self.protocol_path) +")"
        return rep
    
    def display_hierarchy(self, compact=False):
        """Display the names of the different substrates and days present in the experiment instance."""

        if compact == False:
            print(f"{self}:")
            for s in self.substrates_refs:
                print(f"\t{s}")
                for d in s.days_refs:
                    print(f"\t\t{d}")

        elif compact == True:
            print(f"{self.name}:")
            for s in self.substrates_refs:
                print(f"\t{s.name}")
                for d in s.days_refs:
                    print(f"\t\t{d.name}: {d.current_param}")

    def add_susbtrate(self, substrate_name):
        """Add a substrate to the Exp instance created.
        
        Parameters
        ----------
        substrate_name: str
            Name of the substrate to add
        """
        if substrate_name not in self.substrates_names:
            self.substrates_names.append(substrate_name) #update list of names
            substrate = Substrate(substrate_name, parent=self, days=self.days_names) #create Substrate class instance
            self.substrates_refs.append(substrate) #update ref list
    
    def rm_substrate(self, substrate_name):
        """Remove a substrate to from the Exp instance created.
        
        Parameters
        ----------
        substrate_name: str
            Name of the substrate to remove
        """
        if substrate_name in self.substrates_names:
            idx = self.substrates_names.index(substrate_name) #update list of names
            del(self.substrates_names[idx], self.substrates_refs[idx]) #update list of refs

    #Here as is needs to be apply to all experiment
    def add_day(self, day_name):
        """Add a day to the Exp instance created.
        
        Parameters
        ----------
        day_name: str
            Name of the day to add
        """
        if day_name not in self.days_names:
            self.days_names.append(day_name) #update list of names
            for s in self.substrates_refs:
                day = Day(day_name, parent=s) #create Day class instance
                #s.days_names.append(day_name) #update susbtrate class
                s.days_refs.append(day)

    #Here as is needs to be apply to all experiment
    def rm_day(self, day_name):
        """Remove a day from the Exp instance created.
        
        Parameters
        ----------
        day_name: str
            Name of the day to remove
        """
        if day_name in self.days_names:
            idx = self.days_names.index(day_name) #update list of names
            del(self.days_names[idx])
            for s in self.substrates_refs:
                del(s.days_refs[idx]) #update list of refs
    
    def display_dynamic(self, hierarchy: dict, days: List[str], slices: tuple[int, int, int, ChannelNames], image_type: ImageType, ext=None):
        """Display the pictures of selected hives and substrates at requested time. The non-found requests will display empty images.
        
        Parameters
        ----------
        hierarchy: dict
            Dictionnary containg the requested experiment, substrates and hives to use for display. Should be structured as:
                hierarchy = {
                    "ExpName":
                    {
                        "SubstrateName": [0, 1, N] #number of the hive
                    }
                }

        days: List[str]
            List of the requested days written as strings. ex: days=["D1", "D6"]

        slices: Tuple[int, int, int, ChannelNames]
            The slice of interest written as a tuple (site, timepoint, z-stcak, channel)

        image_type: ImageType
            The type of image to use for the display. Cf ImageType documentation to know which ones are available.

        ext: string. (Optionnal)
            The extension of the image. If not set, it will search for standard extension.
        """

        display_montage_organoids_dynamic(hierarchy, [self], days, slices, image_type, ext)

    def read_hierarchy(self, hierarchy):

        """Parse and validate a hierarchy selection dictionary, used to restrict processing to a subset of the experiment.

        Parameters
        ----------
        hierarchy : dict or None
            Dictionary structured as:
            {
                "SubstrateName1": ["D1", ..., "DN"],
                "SubstrateName2": ["D1", ..., "DN"]
            }

        Returns
        -------
        tuple
            (selected_substrates, selected_days_per_substrate)

        Notes
        -----
        - If None, all substrates and days are selected.
        - Used to restrict processing to a subset of the experiment.

        Example
        -------
        >>> hierachy = {
            "A": ["D1", "D6"],
            "B": ["D4", "D8"]
        }
        >>> exp = Exp._from_csv("Filename.csv")
        >>> substrates, days = exp.read_hierarchy()
        >>> print("substrates=", substrates, ", days=", days)
        substrates= ["A", "B"] , days=[["D1", "D6"], ["D4", "D8"]]
        """

        substrates = []
        days = []

        if hierarchy is None:
            substrates = self.substrates_names
            for s in self.substrates_refs:
                days.append(s.days_names)

        else:
            for s_name in list(hierarchy.keys()):
                if any(s_name == elt for elt in self.substrates_names):
                    substrates.append(s_name)
                    s_names = [s.name for s in self.substrates_refs] #substrate names from exp instance
                    s_index = s_names.index(s_name)
                    d_names = [d.name for d in self.substrates_refs[s_index].days_refs] #day names from Susbtrate
                    if any(d_name == elt for elt in d_names for d_name in hierarchy[s_name]):
                        days.append(hierarchy[s_name])
                    else:
                        logger.warning(f"{self.name} > {s_name} | Days from hierarchy {hierarchy[s_name]} do not exist in experiment. --> skipped.\n\tAvailable days are: {d_names}")

        if len(substrates) == 0:
            logger.error(f"{self.name} | Substrates from hierarchy {list(hierarchy.keys())} do not exist in experiment.\n\tAvailable substrate names are: {self.substrates_names}")
            raise ValueError(f"{self.name} | Substrates from hierarchy {list(hierarchy.keys())} do not exist in experiment.\n\tAvailable substrate names are: {self.substrates_names}")
        
        if len(days) == 0:
            logger.error(f"{self.name} | Days from hierarchy do not exist in experiment. ABORTED.")
            raise ValueError(f"{self.name} | Days from hierarchy do not exist in experiment. ABORTED.")
        
        logger.debug(f"Hierarchy sub-set updated: substrate={substrates}; days={days}")

        return substrates, days
    
    def create_montage(self, image_type: ImageType, channel: ChannelNames=None, hierarchy: dict=None, display_hive_number='on'):
        """Save in the experiment path for each day and each substrate, a montage image which is the reconstruction of the entire 
        substrate with image of type 'image_type' and for channel 'channel'.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to operate the montage creation (see Day.create_montage documentation for more information).
        Montages are automatically saved in `<ExperimentPath>` as `Montage_<image_type>_<SubstrateName>_<DayName>_<Channel>.jpeg`.
        
        Parameters
        ----------
        image_type: ImageType
            The type of image to use for substrate reconstruction. cf ImageType documentation to know available types.
        channel: ChannelNames (Optionnal)
            The channel to use for for substrate reconstruction. By default the 'None' value will generate one montage per existing 
            channel. cf ChannelImage documentation to know available channels.
        hierarchy: Dict[List[str]] (Optionnal)
            The sub-set on which you want to perform the substrate reconstruction. By default the 'None' value will run the method 
            on all substrates and days found. If given, hierarchy should be structured as follow:
            ~~~
                hierarchy = {
                    "SubstrateA": ["D0", ..., "DN"],
                    "SubstrateB": ["D0", ..., "DN"]
                }
            ~~~
        display_hive_number: str {'on', 'off'}
            Overlays the hive numbers on the substrate reconstruction image if set to 'on', else doesn't display. By default set to 'on'.

        Notes
        -----
        - Loops over Day instances
        - Automatically save Montage image
        
        """
        
        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)
                s.create_montage(image_type, channel, days=days[idx_s], display_hive_number=display_hive_number)
    
    def crop(self, save_montage: bool=True, hierarchy: dict=None):
        """Perform hive cropping across selected substrates and days and save the cropped images.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to operate the hive cropping (see Hive.crop_hives documentation for more information).
        Cropped images are automatically saved in `<ExperimentPath>/<SusbtrateName>/<DayName>` as
        `Crop_<Filename>.tif`. Saving substrate reconstrauction at each day is enabled trhough the
        save_montage parameter (see ``create_montage`` documentation for more information).

        Parameters
        ----------
        save_montage : bool, default=True
            If True, generate and save crop montage images in Experiment folder as: `Montage_Crop_<Substrate>_<Day>_<Channel>.jpeg`

        hierarchy : dict, optional
            Subset of substrates and days to process. If None, perform hive Should be written as:
            ~~~
                hierarchy = {
                    "SubstrateA": ["D0", ..., "DN"],
                    "SubstrateB": ["D0", ..., "DN"]
                }
            ~~~

        Notes
        -----
        - Orchestrates Substrate → Day → Hive.crop_hive().
        - Crop images are automatically saved.
        - If save_montage set to True, calls Day.create_montage() for execution.
        """

        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)
                s.crop_hives(days=days[idx_s]) #Crop hives
        
        if save_montage == True: #Save Crop montage if requested
            self.create_montage(ImageType.CROP, hierarchy=hierarchy, display_hive_number='on')

    def optimized_crop(self, save_montage: bool=False, hierarchy: dict=None):   
        """Perform optimized hive cropping across the experiment and save the Optimized_Shadow images.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to operate the refine cropping (see ``Hive.optimized_crop_hives`` documentation for more information).
        Optimized cropped images are automatically saved in `<ExperimentPath>/<SusbtrateName>/<DayName>` as
        `Optimized_Shadow_<Filename>.tif`. Saving substrate reconstruction at each day is enabled trhough the
        save_montage parameter (see ``create_montage`` documentation for more information).
        
        Parameters
        ----------
        save_montage: bool. False by default.
            If False, do not create and do not save the associated montages. cf 'create_montage' method for more info.

        hierarchy: dict. None by default.
            If hierarchy is given, will refine crop only on requested substrates and days. The dictionnary should be structured as:
            ~~~
                hierarchy = {
                    "SubstrateA": ["D0", ..., "DN"],
                    "SubstrateB": ["D0", ..., "DN"]
                }
            ~~~
        Notes
        -----
        - Orchestrates Substrate → Day → Hive.optimized_crop_hive().
        - Automatically saves Optimized_Shadow images
        - If save_montage set to True, calls Day.create_montage() for execution 
        """

        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)                
                s.optimized_crop_hives(days=days[idx_s]) #Optimized Crop hives
                if save_montage == True: #Save Crop montage if requested
                    s.create_montage(ImageType.OPTIMIZED_SHADOW, days=days[idx_s])

    def threshold(self, display='off', verbose='off', hierarchy: dict=None, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):
        """Compute thresholds accross selected susbtrates and days.
        
        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to operate threshold computation on Optimized_Shadow pictures. The threshold computation 
        method depends on the channel (see ``Hive.determine_threshold`` documentation for more 
        information). The ``display`` and ``verbose`` parameters enables visual inspection of the 
        threshold computation.
        
        Parameters
        ----------
        display: str. 'off' by default.
            If set to 'on', create for each susbtrate and day the plots related to threshold determination.
            Each subplot corresponds to one hive of the associated day and susbtrate and is composed of: the
            histogram of the pixel values of the hive, the threshold values.       
        verbose: str. 'off' by default.
            print in terminal the logs of this method.       
        hierarchy: dict. None by default.
            If hierarchy is given, will refine crop only on requested substrates and days. The dictionnary should be structured as:
            ~~~
                hierarchy = {
                    "SubstrateA": ["D0", ..., "DN"],
                    "SubstrateB": ["D0", ..., "DN"]
                }
            ~~~

        Notes
        -----
        - Orchestrates Substrate → Day → Hive.threshold_hives().
        - Final computation performed in Hive (Otsu or custom depending on channel).
        - Requires Optimized_Shadow pictures existence.
        """
        
        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)                
                s.threshold_hives(display=display, verbose=verbose, days=days[idx_s], img_type=img_type) #Threshold organoid

    def get_organoid_mask(self, channel: ChannelNames=None, hierarchy: dict=None, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Generate organoid masks and overlay accross selected susbtrates and days.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to generate and save the image of the mask of the organoid and the overlay image. 
        Optimized_Shadow picture (in gray) with the mask (in red), for each Optimized_Shadow image 
        (see Hive.get_organoid_mask documentation for more information). The masks and overlay pictures
        are automatically saved in the propper `<ExpPath>/<SubstrateName>/<DayName>` folder as 
        `Mask_<filename>.tif` and `Overlay_<filename>.tif`.
        
        Parameters
        ----------
            channel: ChannelNames. None by default.
                By default run method on all channel found. If channel is set, will run only on the 
                requested channel. 

            hierarchy: dict. None by default.
                If hierarchy is given, will refine crop only on requested substrates and days. The 
                dictionnary should be structured as:
                    hierarchy = 
                    {
                        "SubstrateA": ["D0", ..., "DN"],
                        "SubstrateB": ["D0", ..., "DN"]
                    }
        
        Notes
        -----
        - Orchestrates Substrate → Day → Hive.generate_organoid_mask().
        - Actual segmentation performed at Hive level.
        - Automatically saves Masks and Overlay pictures
        - Requires Optimized_Shadow pictures existence.
        """

        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)                
                s.generate_organoid_mask(channel=channel, display='off', verbose='off', days=days[idx_s], img_type=img_type) #Organoid organoid

    def manual_discard(self, hierarchy: dict=None):

        """Launch manual discard interface across selected substrates and days.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to operate the manual discard method. For each Ovelray Montage found, display the overlay 
        montage: user may discarded organoids by right-clicking on them (a cyan cross now overlays 
        the organoid). The organoid may be un-discarded using the same method, ie by right-clicking
        again on them (the cyan cross diseapears). Once discarded is done, close the window to create
        a `UpdatedResults.csv` file that will containg `True` value in the `Manually Discarded` 
        column of the discarded organoids.
        
        Parameters
        ----------
            hierarchy: dict. None by default.
                If hierarchy is given, will refine crop only on requested substrates and days. The 
                dictionnary should be structured as:
                    hierarchy = 
                    {
                        "SubstrateA": ["D0", ..., "DN"],
                        "SubstrateB": ["D0", ..., "DN"]
                    }

        Notes
        -----
        - Delegates to Substrate → Day → Hive.manual_discard().
        - Automatically save a jpg image of the discarded hives.
        - Automatically generates a `UpdatedResults.csv`
        - Requires the existence of the Overlay montage
        """

        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)             
                s.manual_discard(days=days[idx_s]) #Organoid

    def get_save_organoid_data(self, properties=OrganoidProperties.rg_properties.value, additionnal_props=OrganoidProperties.additionnal_props.value, \
                               metadata=OrganoidProperties.metadata.value, days: List[str]=None,  hierarchy: dict=None, ResultFile="Results.csv", img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Extract and save organoid measurements across selected susbtrates and days.

        This method loops overall susbtrates and days instances that are declared in ``hierarchy``
        to extract and save organoid data. For each Optimized_Shadow and Mask pictures found, 
        extract for each organoid all metrics requested in ``properties``, ``additionnal_props`` and 
        ``metadata`` (see ``Hive.get_save_organoid_data()`` for more information on extracted metrics). 
        Then, aggregate all extracted metrics from organoids at Day level and store it in 
        `<ExpPath/<SubstrateName>/<DayName>` folder as `Results.csv` (see ``Day.save_data()`` for more 
        information on result aggregation and saving).
        
        ```Note that the mask can be manually modified before running this method to get more accurate
        measurements.
        ```
        
        Parameters
        ----------
        properties: OrganoidProperties.rg_properties.value.
            List of the properties that can directly be recovered from `sckimage.measure.regionsprops`
            and that will be stored in `ResultFile.csv`.
        additionnal_props=OrganoidProperties.additionnal_props.value
            List of extra properties that are extracted from `sckimage.measure.regionsprops` using 
            `extra` parameter and that will be stored in `ResultFile.csv`.
        metadata=OrganoidProperties.metadata.value
            List of metadata that will be stored in `ResultFile.csv`.
        hierarchy: dict. None by default.
            If hierarchy is given, will refine crop only on requested substrates and days. The 
            dictionnary should be structured as:
                hierarchy = 
                {
                    "SubstrateA": ["D0", ..., "DN"],
                    "SubstrateB": ["D0", ..., "DN"]
                }
        ResultFile: str. `ResultFile.csv` by default.
            Name of the csv file in which you want to store the data.

        Notes
        -----
        - Aggregates results at Day level: Day.save_result()
        - Hive performs features extraction: Hive.get_save_organoid data()
        - Automatically generates one result file per day and substrate
        - Requires the existence of Mask and Optimized_Shadow pictures.
        """

        substrates, days = self.read_hierarchy(hierarchy)
        for s in self.substrates_refs:
            if s.name in substrates:
                idx_s = substrates.index(s.name)                
                s.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, days=days[idx_s], ResultFile=ResultFile, img_type=img_type) #Organoid organoid
  
    @classmethod
    def _from_csv(cls, csv_filename):
        """Reconstruct an Exp object from a CSV configuration file.
        
        Parameters
        ----------
        csv_filename: str.
            Full path and name to the csv file.

        CSV file description
        --------------------
        **Path: str.**
            The path leading to your experiment

        **Pattern: str.**
            The pattern used to name the images. ExpName will be replaced by the value entered in `ExpName`, `Substrate`
            will be replaced by the values enetered in `Substrates` column, and `Day` will be replaced by the value entered
            in `Days` column.
        
        **ExpName: str.**
            Name of the experiment. Should be the same one than the one used to replace `ExpName` from `Pattern` field.

        Then, the table has multiple columns that correpond to aquisition parameters and each row is the imaging session of a 
        specific substrate at a specific time.

        **Substrates: str.**
            Column containg the name of the substrates. Note that each value entered in `Substrate` column should correspond to
            a Substrate folder located in `Path`
        
        **Days: str.**
            Column containg the name of the days. Note that each value entered in `Day` column should correspond to
            a Day folder located in `Path\\Substrates`.
        
        **Design: str. {"H2M100", "H2M150", "H2M200", "H2M150b"}**
            Column containg the name of the design.

        **Dimension: str. {"H2M100", "H2M150", "H2M200", "H2M150b"}**
            Column containg the dimension of the original images as HeightxWidth (in pixel).
        
        **Angle: int**
            Column containg the approxiamte rotation angle of the original images (+/- 5°).

        **Aquisition order: str. {"bottom", "bottom-flip"}**
            Column containg the mode of aquisition order.

        **Microscope: str. {"Leica", "Zeiss-Incubator", "Zeiss-Confocal"}**
            Column containg the microscope used during imaging.   

        **Magnification: str. {"x5", "x10", "x20", "x40"}**
            Column containg the magnification used during imaging.     

        CSV file example
        -----------------
        csv_filename is csv file structure as follow:
        ```
            Path    Users/ChimieENS/Documents/Layla/Data/250703-Exp001-B    
            Pattern ExpName_Substrate_Day_0
            ExpName Exp001-B

            Substrates  Days  Design    Dimension   Angle   Aquisition order    Microscope  Magnification
            CondA       D1      H2M200  1944x2592   0       bottom-flip         Leica       x5
            CondA       D4      H2M200  1944x2592   0       bottom              Leica       x5
            CondB       D1      H2M150b 1944x2592   0       bottom-flip         Leica       x5
            CondB       D4      H2M150b 1944x2592   0       exp001              Leica       x5
        ```
        
        """

        if os.path.isfile(csv_filename) == False:
            logger.error(f"Input csv file not found : {csv_filename}")
            raise ValueError(f"{csv_filename} not found.")

        user_inputs = {"Params": []}
        headers = []
        f = open(csv_filename, 'r', newline='', encoding='utf-8-sig')
        reader = csv.reader(f, quoting=csv.QUOTE_NONE)
        for r, row in enumerate(reader):
            if "".join(row) == "":
                idx_empty_row = r
                break
        f.seek(0) #go back at begining of document
        for r, row in enumerate(reader):
            if r < idx_empty_row:
                user_inputs.update({row[0]: row[1]})
            elif r == idx_empty_row + 1:
                for col in row:
                    headers.append(col)
                    user_inputs.update({col: []})
            elif r > idx_empty_row + 1:
                user_inputs["Params"].append({})
                for c, col in enumerate(row):
                    user_inputs["Params"][-1].update({headers[c]: col})
                    user_inputs[headers[c]].append(col)
        substrate_names = list(set(user_inputs["Substrates"]))
        substrate_names.sort()
        day_names = list(set(user_inputs["Days"]))
        day_names.sort()
        user_inputs.update({"Substrate Names": substrate_names, "Day Names": day_names})

        return cls(user_inputs["ExpName"], user_inputs["Path"], user_inputs["Day Names"], user_inputs["Substrate Names"], user_inputs["Pattern"], user_inputs["Params"])
            
class Substrate:

    days_refs: list[Day] #enable autocompletion and will hold Day instances

    def __init__(self, name: str, from_parents: dict, hexagon: dict, days: List[str], params: dict):

        """Initialize a substrate (experimental condition).

        Parameters
        ----------
        name : str
            Substrate name.

        parent : Exp
            Exp instance that called initialization of this
            Substrate instance.

        days : list of str
            Day identifiers.

        params : dict
            Acquisition parameters.

        Attributes
        ----------
        name: str
            Substrate name.
        
        parent : Exp
            Parent experiment (ie, reference to Exp class)

        days : list of str
            Day identifiers.

        params : dict
            Acquisition parameters.

        days_ref: list of Day
            List of Day instances created during initialization.

        Notes
        -----
        - Automatically creates Day instances.
        - Day automatically creates Hive instances.
        - Store reference of Exp class which called it's creation
        - Store references of the generated Day class in a list.
        """

        logger.debug(f"{from_parents["exp_name"]} > {name} | Creating Susbtrate")

        self.name = name
        #self.parent = parent #enable autocompletion and will hold Substrate instances
        self.days_names = days.copy()
        self.params = params
        self.days_refs = []
        self.from_parents = from_parents
        self.hexagon = hexagon
        for_child = from_parents.copy()
        for_child.update({"substrate_name": name,
                          "substrate_path": from_parents["exp_path"] + PATH_DELIMITER + name + PATH_DELIMITER})
        
        #Verify day path existence
        for d in days:
            d_path = for_child["substrate_path"] + d
            if os.path.isdir(d_path) == False:
                logger.warning(f"{from_parents["exp_name"]} > {self.name} | Day folder do not exist: day={d}, path={d_path}.--> Removed from Day instances.\n\tVerify day name in folder and input csv file: they should be identical")
                self.days_names.remove(d)

        #Create Day instances
        for d in self.days_names:
            #Select aquisition parameters
            current_param = None
            for row in self.params:
                if row["Substrates"] == self.name and row["Days"] == d:
                    current_param = row
                    break
            if current_param != None:
                day = Day(d, for_child, hexagon, current_param)
                self.days_refs.append(day)

    def __repr__(self):
        """Defines how to print the Substrate class: Substrate(SubstrateName, ExpName, [D1, ..., DN])"""
        days = ",".join(self.days_names)
        rep = "Substrate(" + self.name + ", " + self.from_parents["exp_name"] + ", " + days + ")"
        return rep

    # def display_hives_dynamic(self, hierarchy: dict, days: list, image_type: ImageType, ext=None):
    #     """Display the pictures of selected hives and substrates at requested times. The non-found requests will display empty images.
        
    #     Parameters
    #     ----------
    #     hierarchy: dict
    #         Dictionnary containg the requested experiment, substrates and hives to use for display. Should be structured as:
    #             hierarchy = {
    #                 "ExpName":
    #                 {
    #                     "SubstrateName": [0, 1, N] #number of the hive
    #                 }
    #             }

    #     days: List[str]
    #         List of the requested days written as strings. ex: days=["D1", "D6"]

    #     slices: Tuple[int, int, int, ChannelNames]
    #         The slice of interest written as a tuple (site, timepoint, z-stcak, channel)

    #     image_type: ImageType
    #         The type of image to use for the display. Cf ImageType documentation to know which ones are available.

    #     ext: string. (Optionnal)
    #         The extension of the image. If not set, it will search for standard extension.
    #     """
    #     display_montage_organoids_dynamic(hierarchy, [self.parent], days, image_type, ext)

    def create_montage(self, image_type: ImageType, channel: ChannelNames=None, days: List[str]=None, display_hive_number='on'):
        """Save in the experiment path for selected days, a montage image which is the reconstruction of the entire substrate for images
        `image_type` and for channel `channel`.
        
        Parameters
        ----------
        image_type: ImageType
            The type of image to use for substrate reconstruction. cf ImageType documentation to know available types.
        channel: ChannelNames (Optionnal)
            The channel to use for for substrate reconstruction. By default the 'None' value will generate one montage per existing 
            channel. cf ChannelImage documentation to know available channels.
        hierarchy: Dict[List[str]] (Optionnal)
            The sub-set on which you want to perform the substrate reconstruction. By default the 'None' value will run the method 
            on all substrates and days found.
        display_hive_number: str {'on', 'off'}
            Overlays the hive numbers on the substrate reconstruction image if set to 'on', else doesn't display. By default set to 'on'.
        days: list of str
            List of days to process. By default, None, meaning it will run on all days.
        
        """
        
        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.create_montage(image_type, channel, display_hive_number=display_hive_number)
            else:
                d.create_montage(image_type, channel, display_hive_number=display_hive_number)

    def crop_hives(self, days: List[str]=None):
        """Crop all original pictures for selected days and automatically saves the Crop pictures.

        This method loops overall days instances that are declared in ``days``
        to operate the hive cropping (see Hive.crop_hives documentation for more information).
        Cropped images are automatically saved in ``<ExperimentPath>/<SusbtrateName>/<DayName>`` as
        ``Crop_<Filename>.tif``. Saving substrate reconstrauction at each day is enabled trhough the
        save_montage parameter (see ``create_montage`` documentation for more information).
        
        Parameters
        ----------
        days : list of str, optional
            Days to process. By default, None, meaning it will run in all Day instances found.

        Notes
        -----
        - Orchestrates  Day → Hive.crop_hive().
        - Automatically saves Crop images.
        - Day executes slicing loops.
        - Hive executes actual cropping.
        """
        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.crop_hives()
            else:
                d.crop_hives()
    
    def optimized_crop_hives(self, n_slice: int=15, k_step: int=10, days: List[str]=None):
        """Perform optimized cropping for selected days.

        This method loops overall selected days instances that are declared in ``hierarchy``
        to operate the refine cropping (see ``Hive.optimized_crop_hives`` documentation for more information).
        Optimized cropped images are automatically saved in `<ExperimentPath>/<SusbtrateName>/<DayName>` as
        `Optimized_Shadow_<Filename>.tif`. Saving substrate reconstruction at each day is enabled trhough the
        save_montage parameter (see ``create_montage`` documentation for more information).

        Parameters
        ----------
        n_slice : int. Default=15
            number of strips.
        k_step : int. Default=10
            width of strip (in pixel)
        days : list of str, optional
            Days to process.

        Notes
        -----
        - Orchestrates  Day → Hive.optimized_crop_hive().
        - Automatically saves Optimized_Shadow pictures.
        - Requires existence of Crop images.
        """

        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.optimized_crop_hives(n_slice=n_slice, k_step=k_step)
            else:
                d.optimized_crop_hives(n_slice=n_slice, k_step=k_step)

    def threshold_hives(self, display='off', verbose='off', days: List[str]=None, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):
        """Compute thresholds for hives in selected days.

        This method loops overall selected days instances that are declared in ``hierarchy``
        to operate threshold computation on Optimized_Shadow pictures. The threshold computation 
        method depends on the channel (see ``Hive.determine_threshold`` documentation for more 
        information). The ``display`` and ``verbose`` parameters enables visual inspection of the 
        threshold computation.
        
        Parameters
        ----------
        display: str. 'off' by default.
            If set to 'on', create for each susbtrate and day the plots related to threshold determination. Each subplot corresponds to one hive of the associated day and susbtrate
            and is composed of: the histogram of the pixel values of the hive, the threshold values.       
        verbose: str. 'off' by default.
            print in terminal the logs of this method.       
        days: List[str]. None by default.
            List of days on which the method will run. By default, None, meaning it will run on all days.

        Notes
        -----
        - Orchestrates  Day → Hive.theshold_hive().
        - Requires existence of Optimized_Shadow images.
        """
        
        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.threshold_hives(display=display, verbose=verbose, img_type=img_type)
            else:
                d.threshold_hives(display=display, verbose=verbose, img_type=img_type)

    def generate_organoid_mask(self, channel: ChannelNames=None, display='off', verbose='off', days: List[str]=None, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Generate organoid masks and overlays for selected days.
        
        This method loops overall selected days instances that are declared in ``hierarchy``
        to generate and save the image of the mask of the organoid and the overlay image. 
        Optimized_Shadow picture (in gray) with the mask (in red), for each Optimized_Shadow image 
        (see Hive.get_organoid_mask documentation for more information). The masks and overlay pictures
        are automatically saved in the propper `<ExpPath>/<SubstrateName>/<DayName>` folder as 
        `Mask_<filename>.tif` and `Overlay_<filename>.tif`.
        
        Parameters
        ----------
            channel: ChannelNames. None by default.
                By default run method on all channel found. If channel is set, will run only on the requested channel. 
            display: str. {'on', 'off'}. 'off' du default.
                For each day, will generate a window with display for each hive, the different image processing steps.
            verbose: str. {'on', 'off'}. 'off' du default.
                Prints the logs of the method in the terminal.
            days: List[str]. None by default.
                List of days on which the method will run. By default, None, meaning it will run on all days.

        Notes
        -----
        - Orchestrates Day → Hive.generate_organoid_mask().
        - Requires existence of Optimized_Shadow images.
        - Requires computated thresholds.
        - Automatically save organoid mask and overlay.
        """

        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.generate_organoid_mask(channel=channel, display=display, verbose=verbose, img_type=img_type)
            else:
                d.generate_organoid_mask(display=display, verbose=verbose, img_type=img_type)

    def manual_discard(self, days: List[str]=None):

        """Launch manual discard interface across selected days.

        This method loops overall days instances that are declared in ``hierarchy``
        to operate the manual discard method. For each Ovelray Montage found, display the overlay 
        montage: user may discarded organoids by right-clicking on them (a cyan cross now overlays 
        the organoid). The organoid may be un-discarded using the same method, ie by right-clicking
        again on them (the cyan cross diseapears). Once discarded is done, close the window to create
        a `UpdatedResults.csv` file that will containg `True` value in the `Manually Discarded` 
        column of the discarded organoids.
        
        Parameters
        ----------
        days: List[str]. None by default.
            List of days on which the method will run. By default, None, meaning it will run on all days.
        """

        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.manual_discard()
            else:
                d.manual_discard()

    def get_save_organoid_data(self, properties=OrganoidProperties.rg_properties.value, additionnal_props=OrganoidProperties.additionnal_props.value, \
                               metadata=OrganoidProperties.metadata.value, ResultFile="Result.csv", days: List[str]=None, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Extract and save organoid data for selected days.
    
        This method loops overall days instances that are declared in ``hierarchy``
        to extract and save organoid data. For each Optimized_Shadow and Mask pictures found, 
        extract for each organoid all metrics requested in ``properties``, ``additionnal_props`` and 
        ``metadata`` (see ``Hive.get_save_organoid_data()`` for more information on extracted metrics). 
        Then, aggregate all extracted metrics from organoids at Day level and store it in 
        `<ExpPath/<SubstrateName>/<DayName>` folder as `Results.csv` (see ``Day.save_data()`` for more 
        information on result aggregation and saving).
        
        ```Note that the mask can be manually modified before running this method to get more accurate
        measurements.
        ```
        
        Parameters
        ----------
        properties: OrganoidProperties.rg_properties.value.
            List of the properties that can directly be recovered from 'sckimage.measure.regionsprops' and that will be stored in 'ResultFile.csv'.
        additionnal_props=OrganoidProperties.additionnal_props.value
            List of extra properties that are extracted from 'sckimage.measure.regionsprops' using 'extra' parameters and that will be stored in 'ResultFile.csv'.
        metadata=OrganoidProperties.metadata.value
            List of metadata that will be stored in 'ResultFile.csv'.
        ResultFile: str. 'ResultFile.csv" by default.
        days: List[str]. None by default.
            List of days on which the method will run. By default, None, meaning it will run on all days.

        Notes
        -----
        - Aggregates Day-level data
        - Data extraction at Hive-level:
        """

        for d in self.days_refs:
            if days is not None:
                if d.name in days:
                    d.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, ResultFile=ResultFile, img_type=img_type)
                    #d.save_organoid_data(data, headers=properties+additionnal_props+metadata, ResultFile=ResultFile)
            else:
                d.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, ResultFile=ResultFile, img_type=img_type)
                #d.save_organoid_data(data, headers=properties+additionnal_props+metadata, ResultFile=ResultFile)

class Day:

    hives_refs: list[Hive] #enable autocompletion and will hold Day instances

    def __init__(self, name, from_parents, hexagon, current_param):

        logger.debug(f"{from_parents["exp_name"]} > {from_parents["substrate_name"]} > {name} | Creating Day")

        self.name = name
        self.current_param = current_param
        self.hives_refs = []
        self.hive_number = [] #will hold the static number (ie depending on aquisition parameter)
        self.image_number = [] #will hold the image number 
        self.img_original_dim = None
        self.img_crop_dim = None
        self.img_optimizedshadow_dim = None
        self.img_overlay_dim = None
        self.img_mask_dim = None
        self.hexagon = hexagon
        self.from_parents = from_parents
        self.path = from_parents["substrate_path"] + self.name + PATH_DELIMITER

        #Store filenames if exist, else empty list
        self.filenames_original, self.image_number = self.get_filenames(ImageType.ORIGINAL.value, current_param["Microscope"])
        self.filenames_crop, _ = self.get_filenames(ImageType.CROP.value, current_param["Microscope"])
        self.filenames_optimized_shadow, _ = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value, current_param["Microscope"])
        self.filenames_mask, _ = self.get_filenames(ImageType.MASK.value, current_param["Microscope"])
        self.filenames_overlay, _ = self.get_filenames(ImageType.OVERLAY.value, current_param["Microscope"])
        self.filenames_core_mask, _ = self.get_filenames(ImageType.CORE_MASK.value, current_param["Microscope"])
        self.filenames_shell_mask, _ = self.get_filenames(ImageType.SHELL_MASK.value, current_param["Microscope"])

        if self.filenames_original is None:
            # raise UserWarning(f"{self.path} do not contain any usable picture.")
            #warnings.warn(f"{self.path} do not contain any usable picture.", UserWarning)
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Not usable pictures found --> skipped.\nVerify image name fits pattern.")
            return

        #Verify images have consistent dimensions
        for ty in ImageType:
            if ty is not ImageType.MONTAGE:
                if len(self.get_attribute_name(ty, 'filenames_', '')) > 0:
                    self.check_img_dims(ty)

        #Assign parameters (hive real size and pixel ratio)
        self.hive_height, self.n_expect, self.um2pxl = uf.assign_dim(self.current_param["Design"], \
            self.current_param["Microscope"], self.current_param["Magnification"], self.current_param["Dimension"])
        # self.hive_side = HexagonMath.get_side_from_height(self.hive_height) #round(self.hive_height/(2*np.sin(np.pi/3)))
        # self.hive_diag = HexagonMath.get_diagonal_from_height(self.hive_height) #2*self.hive_side

        #Create associated hive mask
        hexagon_param = (self.current_param["Design"],self.current_param["Microscope"],self.current_param["Magnification"],self.current_param["Dimension"])
        if hexagon_param in self.hexagon.keys():
            self.hive_mask = self.hexagon[hexagon_param]["hexagon_full_scale"]
            self.hive_mask_rs = self.hexagon[hexagon_param]["hexagon_rescale"]
            current_hexagon = hexagon[hexagon_param]
        else:
            current_hexagon = None

        #Generate Hive Class
        for_child = from_parents.copy()
        for_child.update({"day_name": self.name, "day_path": self.path, 
        "filenames_original": self.filenames_original, "filenames_crop": self.filenames_crop,
        "filenames_overlay": self.filenames_overlay, "filenames_mask": self.filenames_mask,
        "filenames_optimized_shadow": self.filenames_optimized_shadow,
        "um2pxl": self.um2pxl, "img_original_dim": self.img_original_dim})
        #idx = uf.idx_acquisition_order(self.current_param["Aquisition order"])
        for i,file in enumerate(self.filenames_original):
            self.hive_number.append(uf.get_hive_number(self.image_number[i], self.current_param["Aquisition order"]))
            hive = Hive(self.hive_number[-1], self.image_number[i], for_child, current_hexagon, file)
            self.hives_refs.append(hive)
  
    def __repr__(self):
        """Defines how to print the Day class: Day(DayName, SubstrateNmae, ExpName, Aquisition Parameters)"""
        rep = "Day(" + self.name + ", " + self.from_parents["substrate_name"] + ", " +  repr(self.current_param) + ")"
        return rep
    
    def get_attribute_name(self, image_type: ImageType, prefixe: str, suffixe: str):
        """Return the requested attribute containing one ImageType in its name: prefixe + image_type + suffixe
        
        Parameters
        ----------
        image_type: ImageType.
            The image type name present in the attribute: it will use the lower case name of the parameter.
        prefixe: str.
            The prefixe of the attribute (ie, before the image_type)
        suffixe: str.
            The prefixe of the attribute (ie, after the image_type)

        """
        attr_name = f"{prefixe}{image_type.name.lower()}{suffixe}"
        return getattr(self, attr_name)
    
    def get_slices_range(self):
        """Return the slices range to loop over: ([sites], [timepoints], [zs], [channels])"""
        
        #Sort channels to get BF as 1st one in list
        chans = list(self.filenames_original[0][0][0][0].keys())
        idx_BF = chans.index(ChannelNames.BF.name)
        if idx_BF != 0 :
            del chans[idx_BF]
            chans.insert(ChannelNames.BF.name, 0)

        slices = (
            range(len(self.filenames_original[0])), #number of sites
            range(len(self.filenames_original[0][0])), #number of timepoints
            range(len(self.filenames_original[0][0][0])), #number of z-stack
            list(chans) #channels
        )

        return slices
    
    def check_img_dims(self, image_type: ImageType):
        """Verify the images of type image_type have similar dimensions and store dimension in attribute self.dim_name"""

        dims = []
        # folder_path = self.parent.parent.path + PATH_DELIMITER + self.parent.name + PATH_DELIMITER + self.name
        filenames = self.get_attribute_name(image_type, 'filenames_', '')
        filenames_flat = uf.flatten_list_recursive(filenames, final='dict') 
        for files in filenames_flat:
            with Image.open(self.path + PATH_DELIMITER + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            logger.error(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | {image_type.name}  images don't have same dimension")
            raise RuntimeError(f"{image_type.name}  images don't have same dimension in {self.path}")
        elif len(set(dims)) == 1:
            dim_name =f"img_{image_type.name.lower()}_dim"
            setattr(self, dim_name, (dims[0][1], dims[0][0]) )
        else:
            logger.error(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | No images {image_type.name} found.")
            raise RuntimeError(f"{self.path} issue, no {image_type.name} images found")
    
    def get_filenames(self, image_type: ImageType, microscope: str):
        """Return filenames structured as: filenames[hive][site][time][height][region] = {ChanName1: filename1, ChanName2: filename2}
        
        Parameters
        ----------
        image_type: ImageType
            Image type you want to search the filenames. Cf ImageType documentation for information on available values.
        microscope: str. {"Leica", "Zeiss-Incubator", "Zeiss-Confocal"}
            Microscope used for the imaging session of this specific substrate and day.

        Returns
        -------
        filenames: list.
            The filenames of type image_type for the imaging session of this sepcific day and substrate. It is composed of nested lists 
            with ended up by dictionnary. By levels there are: 1)hive, 2)sites, 3)timepoint 4)height. The final dictionnary has the channel 
            name as key and the filename as value: filenames[hive][site][time][height][region] = {ChanName1: filename1, ChanName2: filename2}. 
            For example, if the imaging session contains is a simple one with only one picture per hive for two hives in BF and CY3, no 
            timelapse and no z-stack:
            ~~~
            filenames = [
                [   #Hive 1
                    [   #Site 1
                        [   #Timepoint 1
                            [   #Height 1 
                                {"BF": "hive1_BF.tif", "CY3": "hive1_CY3.tif"} #Channels & Filenames
                            ]
                        ],
                    #Hive 2
                    [   #Site 1
                        [   #Timepoint 1
                            [   #Height 1 
                                {"BF": "hive2_BF.tif", "CY3": "hive2_CY3.tif"} #Channels & Filenames
                            ]
                        ]
                    ]
                ]
            ]
            ~~~
        image_number: List[int].
            List of same size the main filenames list (ie, for hive) that contains the number in the filename corresponding to the hive position.
        """
        
        filenames = None
        image_number = None

        filename_pattern = self.from_parents["pattern"]
        path = self.path
        exp_name = self.from_parents["exp_name"]
        substrate_name = self.from_parents["substrate_name"]
        day = self.name

        if microscope == "Leica":
            filenames, image_number = uf.get_filenames_leica(filename_pattern, path, exp_name, substrate_name, day, image_type)
        
        elif microscope == "Zeiss-Incubator":
            filenames, image_number = uf.get_filenames_zeiss_incub_pos(path, image_type)

        elif microscope == "Zeiss-confocal":
            filenames, image_number = uf.get_filenames_zeiss_confocal_czi(image_type)

        else:
            #warnings.warn("Not implemented yet.")
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | {microscope} not implemented yet.")

        return filenames, image_number
    
    def adjust_filename_pattern(self, microscope: str, pattern: str, image_type: ImageType, params: dict):

        """Returns the updated filename pattern for this specific subatrate and day.
        
        Parameters:
        -----------
        microscope: str. {"Leica", "Zeiss-Incubator", "Zeis-Confocal"}.
            The microscope used for the imaging session.
        pattern: str.
            Pattern to seach in filenames.
        image_type: ImageType.
            Type of image.
        params: dict.
            Parameters required to update pattern

        Returns:
        --------
        updated_pattern: str
            updated pattern based on the aquisition parameters, specific substrate and day.
        """

        if microscope == "Leica":
            updated_pattern = uf.adjust_filename_pattern_leica(pattern, image_type, params)
        
        elif microscope == "Zeiss-Incubator":
            updated_pattern = uf.adjust_filename_pattern_zeiss_incub_pos(pattern, image_type, params)

        return updated_pattern
    
    @log_method()
    def find_rotation_angle(self, SCALE_RATIO, slices):
        """Return the rotation angle of the hives for the current day and substrate.
        
            For each hive of the current substrate and day, determine the rotation
            angle within the range ±5° from ``self.current_param["Angle"]`` (see
            ``Hive.find_hive_rotation_angle`` for more information). The rotation
            angle of the imaging session is then defined as the median of the
            rotation angles determined for each hive.

            Parameters
            ----------
            SCALE_RATIO : int
                Factor applied to rescale original images and hive masks.
                This factor is usually set to 10 and is used to speed up the
                convolution performed during the determination of each hive's
                rotation angle.

            slices : Tuple[int, int, int, ChannelNames]
                Slice on which the method should be run. This method runs only
                on the BF image. The tuple contains:
                - site (int): Position number within the hive.
                - timepoint (int): Timepoint number.
                - Z (int): Height number.
                - channel (ChannelNames): Name of the channel. Must always be BF
                for this method.

            Returns
            -------
            best_angle : float
                Median of the rotation angles (in degrees) for the imaging session.
        """
        
        logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Computing best angle")
        
        best_angles = []
        for i, hive in enumerate(self.hives_refs):
            best_angles.append(hive.find_hive_rotation_angle(self.hive_mask_rs, int(self.current_param["Angle"]), 5, 0.5, SCALE_RATIO, slices)) #Compute & store best angle
        best_angle = np.nanmedian(best_angles) #select best angle from set
        
        return best_angle

    @log_method()
    def create_montage(self, image_type: ImageType, channel: ChannelNames=None, slices: Tuple[int,int,int,ChannelNames]=None, display_hive_number='on', SCALE_RATIO: int=6):
        """ Create and save a 5x5 montage image of all hives for the selected slices.

        The method reconstructs a montage image from individual hive images
        acquired during the imaging session. Missing images are replaced by
        empty placeholders. The montage layout follows the predefined
        substrate organization:

            - Column 1: 3 hives
            - Column 2: 4 hives
            - Column 3: 5 hives
            - Column 4: 3 hives
            - Column 5: 4 hives

        Hive numbers can optionally be displayed on the montage. The final
        image is downscaled, converted to 8-bit, saved as a JPEG file, and
        returned.

        Parameters
        ----------
        image_type : ImageType
            Type of image to use (e.g., raw, cropped, processed).

        channel : ChannelNames, optional
            Channel to process. If None, the channel provided in `slices`
            (or the full slice range) is used.

        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).
            If None, the full slice range obtained from
            ``self.get_slices_range()`` is used.

        display_hive_number : {"on", "off"}, default="on"
            Whether to overlay hive numbers on the montage.

        SCALE_RATIO : int, default=6
            Downscaling factor applied to the montage using local mean
            downsampling. Larger values produce smaller output images.

        Returns
        -------
        numpy.ndarray
            The final 8-bit montage image after resizing and annotation.

        Notes
        -----
        - Missing hive images are automatically replaced by empty images.
        - The montage is arranged according to the acquisition order defined
        in ``self.current_param["Aquisition order"]``.
        - The output image is saved as:
            ``Montage_<image_type><experiment>_<substrate>_<day>_<channel>.jpg``
        """

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Creating montage {image_type.name}")

        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])
        
        if channel is None:
            channel = slices[3]
        else:
            channel = [channel]

        #Image acquisition order:
        idx = uf.idx_acquisition_order(self.current_param["Aquisition order"])

        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for c_i, c in enumerate(channel):

                        #Adjust filename pattern:
                        params = {
                            "exp": self.from_parents["exp_name"], "substrate": self.from_parents["substrate_name"], "day": self.name, 
                            "t": t, "z": z, "c": c_i, "position": None
                        }

                        #Open images
                        filenames = self.get_attribute_name(image_type, "filenames_", "")
                        filenames_flat = [elt[s][t][z][c] for elt in filenames]
                        img = [None for _ in range(len(self.hives_refs))]
                        i=0
                        for hive in self.hives_refs:
                            img[i] = hive.open_image(image_type, (s,t,z,c), rgb=True)
                            i+=1
                        
                        pattern = self.adjust_filename_pattern(self.current_param["Microscope"], self.from_parents["pattern"], image_type.value, params)

                        #If missing pictures, adding empty ones
                        img, filenames = uf.add_empty_img(img, filenames_flat, image_type.value + pattern, 19)

                        #Get coordinate for montage
                        img_sz = img[0].shape
                        img_type = img[0].dtype
                        coord, idx_flat = uf.get_hive_coord_montage(img_sz, idx, base='hive')

                        #Reconstruct picture
                        if len(img_sz) > 2:
                            montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=img_type)
                            for i in range(len(img)):
                                montage[coord[i][1][0]:coord[i][1][1], coord[i][0][0]:coord[i][0][1], :] = img[idx_flat[i]]
                        else:
                            montage = np.empty((img_sz[0]*5, img_sz[1]*5), dtype=img_type)
                            for i in range(len(img)):
                                montage[coord[i][1][0]:coord[i][1][1], coord[i][0][0]:coord[i][0][1]] = img[idx_flat[i]]

                        #Display hive number
                        if display_hive_number =='on':
                            montage_pil = Image.fromarray(montage)
                            draw = ImageDraw.Draw(montage_pil)
                            font = ImageFont.truetype("arial.ttf", size=200)
                            for i, img_number in enumerate(idx_flat):

                                if len(img_sz) > 2:
                                    draw.text((coord[i][0][0]+25, coord[i][1][0]+50), str(uf.get_hive_number(img_number, self.current_param["Aquisition order"])), font=font, fill=(255,255,255))
                                else:
                                    draw.text((coord[i][0][0]+25, coord[i][1][0]+50), str(uf.get_hive_number(img_number, self.current_param["Aquisition order"])), font=font)
                            #montage_pil.show()
                            montage = np.array(montage_pil)
                            montage_type = montage.dtype

                        #Downsize image
                        if len(montage.shape) > 2:
                            montage = downscale_local_mean(montage, (SCALE_RATIO, SCALE_RATIO,1))
                        else:
                            montage = downscale_local_mean(montage, (SCALE_RATIO, SCALE_RATIO))
                        montage = montage.astype(montage_type)

                        #Transform in 8bit
                        montage = img_as_ubyte(montage)

                        #Save montage
                        if image_type is ImageType.CROP:
                            out_path = self.from_parents["exp_path"] + PATH_DELIMITER + \
                            "Montage_" + image_type.value + self.from_parents["exp_name"] + "_" + self.from_parents["substrate_name"] + "_" + self.name + "_" + c + ".jpg"
                        else:
                            out_path = self.path + PATH_DELIMITER + \
                            "Montage_" + image_type.value + self.from_parents["exp_name"] + "_" + self.from_parents["substrate_name"] + "_" + self.name + "_" + c + ".jpg"
                        imsave(out_path, montage)

        logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Montage saved: {out_path}")

        return montage

    @log_method()
    def crop_hives(self, slices: Tuple[int,int,int,ChannelNames]=None):

        """Crop and rotate all hive images for the selected slices.

        For each selected site, timepoint, and z-plane, the method first
        computes the optimal rotation angle using the BF channel. This angle
        is then applied to all channels when cropping each hive image.

        Cropped images are saved to disk. After processing, crop filenames
        are updated and the method verifies that all cropped images share
        the same dimensions. A RuntimeError is raised if inconsistent
        dimensions are detected.

        Parameters
        ----------
        SCALE_RATIO : int
            Downscaling factor used during rotation angle estimation and
            cropping operations.

        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).
            If None, the full slice range obtained from
            ``self.get_slices_range()`` is used.

        Raises
        ------
        RuntimeError
            If cropped images do not all share identical dimensions.

        Attributes Set
        --------------
        filenames_crop : list
            Updated list of cropped image filenames.

        img_crop_dim : Tuple[int, int]
            Dimensions of cropped images as (height, width).

        Notes
        -----
        - The rotation angle is computed using the BF channel only.
        - The same rotation angle is applied to all channels for a
        given (site, timepoint, z) combination.
        - Cropped images are saved automatically via ``hive.crop_hive(..., save=True)``.
        """

        n_hives = len(self.hives_refs)

        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])

        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:

                    #Compute rotation angle
                    best_angle = self.find_rotation_angle(SCALE_RATIO, (s,t,z,ChannelNames.BF.name))

                    for c in slices[3]:

                        #Crop & rotate image
                        for i,hive in enumerate(self.hives_refs):
                            print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Cropping pictures... {i}/{n_hives}", end="\r")
                            hive.crop_hive(SCALE_RATIO, best_angle, slices=(s,t,z,c), save=True)

        #Update Crop filenames
        self.filenames_crop, _ = self.get_filenames(ImageType.CROP.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_crop = hive.update_filename(ImageType.CROP, filenames_from_day=self.filenames_crop)

        #Verify crop pictures have same dimension
        #folder_path = self.parent.parent.path + PATH_DELIMITER + self.parent.name + PATH_DELIMITER + self.name
        dims = []
        for files in uf.flatten_list_recursive(self.filenames_crop):
            with Image.open(self.path + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            logger.error(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Cropped images  don't have same dimension.")
            raise RuntimeError(f"Cropped images don't have same dimension in {self.path}")
        self.img_crop_dim = (dims[0][1], dims[0][0])

        print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Cropping pictures done: {self.img_crop_dim}", end="\r")
        print()

    @log_method()
    def optimized_crop_hives(self, slices: Tuple[int,int,int,ChannelNames]=None, n_slice: int=15, k_step: int=10, display='off'):

        """Refine previously cropped hive images using adaptive hexagonal masks.

        This method performs a secondary cropping step on existing
        cropped images. A sequence of concentric hexagonal side masks is
        generated and applied to each hive to refine shadow removal and
        border alignment.

        Refined images are saved to disk, and filename bookkeeping is updated.

        Parameters
        ----------
        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).
            If None, the full slice range obtained from
            ``self.get_slices_range()`` is used.

        n_slice : int, default=15
            Number of concentric hexagonal masks used during optimization.

        k_step : int, default=10
            Pixel step used to shrink the hexagonal side length between
            successive masks.

        display : {"on", "off"}, default="off"
            Whether to display intermediate optimization results.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If no cropped images are available, the method exits without processing.

        Attributes Set
        --------------
        filenames_optimized_shadow : list
            Updated list of optimized image filenames.
        """
        
        n_hives = len(self.hives_refs)
        hives_mask_side = [None] * n_slice
        side_pxl = round(self.hive_height*self.um2pxl/(2*np.sin(np.pi/3)))
        height_pxl = self.hive_height*self.um2pxl


        #Verify Crop images exist
        if len(self.filenames_crop) == 0:
            #warnings.warn(f"No Crop images found for {self.from_parents["substrate_name"]}-{self.name}")
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | No Crop images found --> skipped.")
            return
        
        #Generate the concentric hexagon sides
        for j,k in enumerate(range(k_step, n_slice*k_step+k_step, k_step)):
            side_k = round((height_pxl - 2*k)/(2*np.sin(np.pi/3)))
            hives_mask_side[j] = HexagonMath.hexagon_outline_ndarray(self.img_crop_dim, [round(self.img_crop_dim[0]/2), round(self.img_crop_dim[1]/2)], side_k, thickness=k_step, return_type='side')
        
        #Select slices
        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])

        #For each slice, optimize crop
        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for c in slices[3]:
                        for j,hive in enumerate(self.hives_refs):
                            print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Refine cropping... {j}/{n_hives}", end="\r")
                            hive.optimized_crop_hive(hives_mask_side, k_step, (s,t,z,c), display=display)

        #Update filenames
        self.filenames_optimized_shadow, _ = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_optimized_shadow = hive.update_filename(ImageType.OPTIMIZED_SHADOW, filenames_from_day=self.filenames_optimized_shadow)

        print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Refine cropping done.       ", end="\r")
        print()

    @log_method()
    def threshold_hives(self, slices: Tuple[int,int,int,ChannelNames]=None, display='off', verbose='off', img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Determine intensity thresholds for organoid segmentation in all hives.

        For each selected slice (site, timepoint, z-plane, channel), this method
        computes a threshold for every hive image in the session. BF channel
        images use a custom threshold, while other channels use Otsu's method.

        Optional visualization displays histograms and selected thresholds
        for each hive.

        Parameters
        ----------
        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).
            If None, the full slice range obtained from
            ``self.get_slices_range()`` is used.

        display : {"on", "off"}, default="off"
            Whether to display histograms and thresholds.

        verbose : {"on", "off"}, default="off"
            Forwarded to ``hive.custom_threshold_hive`` for detailed output.

        Returns
        -------
        list
            Nested list of thresholds structured as:
            ``thresholds[hive][site][timepoint][z][channel]``,
            each element a tuple of two numpy floats.

        Warns
        -----
        UserWarning
            If no optimized-shadow images exist, threshold determination is skipped.

        Notes
        -----
        - Histograms are displayed in a compact grid layout when `display="on"`.
        - Thresholding is performed per hive and per slice.
        """

        #Verify the image exists
        filename_attribute = self.get_attribute_name(img_type, "filenames_", "")
        if len(filename_attribute) == 0:
            #warnings.warn(f"No Optimized_Shadow images found for {self.from_parents["substrate_name"]}-{self.name}")
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | No {img_type} pictures found --> skipped.")
            return
        
        n_hives = len(self.filenames_optimized_shadow)

        #Select slices
        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])

        #Initiate output threshold variable
        thresholds = []
        for hive in self.hives_refs:
            thresholds.append([[[ {} for _ in slices[2] ] for _ in slices[1]] for _ in slices[0]])

        #For each slice
        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for c in slices[3]:

                        #Create figure is display == 'on'
                        if display == 'on':
                            n_r, n_c = uf.get_compact_grid(n_hives) #adequate subplot grid
                            fig, ax = plt.subplots(nrows=n_r, ncols=n_c, sharex=True, sharey=True, 
                                                num=f"Threshold for {self.from_parents["exp_name"]}_{self.from_parents["substrate_name"]}_{self.name}_{c}") #create subplots
                            fig.supxlabel('Pixel value (in range [0;1])')
                            fig.supylabel('Counts')
                            if n_r > 1:
                                idx_ax = list(itertools.product(range(n_r), range(n_c)))
                            else:
                                idx_ax = [elt for elt in range(n_c)]
                        else:
                            fig=None
                            ax=None

                        #Determine threshold
                        for j,hive in enumerate(self.hives_refs):
                            print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Determining threshold... {j}/{n_hives}", end="\r")
                            if display == 'on':
                                current_ax =  ax[idx_ax[j]]
                            else:
                                current_ax = None
                            if c == ChannelNames.BF.name:
                                thresholds[j][s][t][z][c] = hive.custom_threshold_hive((s,t,z,c), fig1=fig, ax1=current_ax, title=f"#{hive.hive_number}", verbose=verbose, img_type=img_type)
                            else:
                                thresholds[j][s][t][z][c] = hive.otsu((s,t,z,c), img_type=img_type)

                        print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Determining threshold done.    ", end="\r")
                        print()

                        if display == 'on':
                            for ax in fig.axes:
                                text = []
                                for txt in ax.texts: text.append(txt)
                                for item in (
                                    ax.title,
                                    ax.xaxis.label,
                                    ax.yaxis.label,
                                    *ax.get_xticklabels(),
                                    *ax.get_yticklabels(),
                                    *text
                                ):
                                    item.set_fontsize(20)
                            plt.show()
        return thresholds

    @log_method()
    def generate_organoid_mask(self, channel: ChannelNames=None, slices: Tuple[int,int,int,ChannelNames]=None, display='off', verbose='off', manual_discard='off', img_type: ImageType=ImageType.OPTIMIZED_SHADOW):
        
        """Generate organoid segmentation masks for selected hives and slices.

        For each selected slice (site, timepoint, z-plane, channel), this method
        computes organoid masks using the optimized shadow-corrected images.
        The segmentation pipeline is implemented at the hive level.

        If no optimized-shadow images exist, the method issues a warning and
        returns immediately.

        Parameters
        ----------
        channel : ChannelNames, optional
            Channels on which to perform segmentation. If None, the channels
            from `slices` or full range are used.

        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).

        display : {"on", "off"}, default="off"
            Whether to display intermediate segmentation steps.

        verbose : {"on", "off"}, default="off"
            Forwarded to ``hive.generate_organoid_mask`` for detailed output.

        manual_discard : {"on", "off"}, default="off"
            If "on", triggers manual discard after mask generation.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If no optimized-shadow images exist, segmentation is skipped.

        Attributes Set
        --------------
        filenames_mask : list
            Updated list of mask image filenames.

        filenames_overlay : list
            Updated list of overlay image filenames.
        """

        #Verify optimized shadow pictures exist
        filename_attribute = self.get_attribute_name(img_type, "filenames_", "")
        if len(filename_attribute) == 0:
            #warnings.warn(f"{self.from_parents["substrate_name"]}_{self.name}, No optimized images found. Cannot generate masks.")
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | No {img_type} found --> skipped.")
            return

        #Select slices
        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])
        if channel is None:
            channel = slices[3]
        else:
            channel = [channel]

        n_hives = len(self.filenames_optimized_shadow)

        #For each slice
        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for c in channel:
        
                        #Create subplot
                        if display == 'on':
                            #1st dim: images
                            #2nd dim: original, custom_threshold, binary operation, filtered, overlay?
                            fig, ax = plt.subplots(ncols=n_hives, nrows=4, num=f"Segmentation Steps - {self.from_parents["exp_name"]}_{self.from_parents["substrate_name"]}_{self.name}_{c}")
                            ax[0, 0].set_ylabel("Original")
                            ax[1, 0].set_ylabel("Threshold")
                            ax[2, 0].set_ylabel("Binary operations")
                            ax[3, 0].set_ylabel("Mask")
                        else:
                            fig=None
                            ax=None

                        #Generate mask & overlay
                        for j,hive in enumerate(self.hives_refs):
                            print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Generating organoid mask... {j}/{n_hives}", end="\r")
                            if display == 'on':
                                current_ax_idx =  j
                            else:
                                current_ax_idx = None
                            hive.generate_organoid_mask((s,t,z,c), fig=fig, ax=ax, ax_idx=current_ax_idx, verbose=verbose, img_type=img_type)

        print(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Generating organoid mask done.    ", end="\r")
        print()

        #Update filenames
        self.filenames_mask = self.get_filenames(ImageType.MASK.value, self.current_param["Microscope"])
        self.filenames_overlay = self.get_filenames(ImageType.OVERLAY.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_mask = hive.update_filename(ImageType.MASK, filenames_from_day=self.filenames_mask)
            hive.filenames_overlay = hive.update_filename(ImageType.OVERLAY, filenames_from_day=self.filenames_overlay)

        #Display
        if display == 'on':
            plt.show()

        if manual_discard == 'on':
            self.manual_discard()

    @log_method()
    def get_save_organoid_data(self, slices: Tuple[int,int,int,ChannelNames]=None, properties=OrganoidProperties.rg_properties.value, \
                               additionnal_props=OrganoidProperties.additionnal_props.value, metadata=OrganoidProperties.metadata.value, ResultFile="Results.csv", img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Retrieve organoid measurements and save them to a CSV file.

        Data are collected from each hive using ``hive.get_organoid_data``.
        Standard, additional, metadata, and fluorescence channel-specific
        properties are aggregated and saved. In the file, each row corresponds
        to a one organoid.

        Parameters
        ----------
        slices : Tuple[int, int, int, ChannelNames], optional
            Slice specification as (site, timepoint, z, channel).

        properties : list of str, default=OrganoidProperties.rg_properties.value
            Standard organoid properties to retrieve that are accesible directly 
            through ``skimage.measure.regionsprops`` method.

        additionnal_props : list of str, default=OrganoidProperties.additionnal_props.value
            Additional organoid properties to include that are accessuble through ``extra``
            parameter of ``skimage.measure.regionsprops`` method.

        metadata : list of str, default=OrganoidProperties.metadata.value
            Metadata fields to include.

        ResultFile : str, default="Results.csv"
            CSV file to save aggregated organoid data.

        Returns
        -------
        list
            Aggregated organoid data from all hives.

        Notes
        -----
        - Fluorescence channel properties are automatically combined.
        - Data are saved via ``self.save_organoid_data``.
        - Prints progress messages for each hive processed.
        - Output file contains the data for all organoid of the current
        substrate and imaging session. Each row corresponds to one organoid
        and the columns contains associated data.
        - see ``Hive.get_organoid_mask`` for more info on extracted data.
        """
        
        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Aggregating organoid data")

        n_hives = len(self.filenames_optimized_shadow)   

        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]]) 

        #Get organoid data
        data = []
        organoid_count = 0
        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for i,hive in enumerate(self.hives_refs):
                        print(f"{self.from_parents["substrate_name"]}_{self.name}, Getting organoid data... {i}/{n_hives}", end="\r")
                        data_hive, organoid_count = hive.get_organoid_data(organoid_count, (s,t,z,slices[3]), properties=properties, additionnal_props=additionnal_props, metadata=metadata, img_type=img_type)
                        data.append(data_hive)
        logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Organoid data aggregated")
        print()

        #Save organoid data
        if len(slices[3]) > 1:
            fluo_chan = slices[3].copy()
            fluo_chan.remove(ChannelNames.BF.name)
            fluo_props = list(itertools.product(OrganoidProperties.fluo_props.value, fluo_chan)) + list(itertools.product(OrganoidProperties.extra_props.value, fluo_chan))
            fluo_props = ["_".join(str(el) for el in elt) for elt in fluo_props]
        else: fluo_props = []
        headers = properties + additionnal_props + fluo_props + metadata
        self.save_organoid_data(data, headers, ResultFile=ResultFile)
        print(f"{self.from_parents["substrate_name"]}_{self.name}, Organoid data saved.    ")

        return data


    def save_organoid_data(self, data, headers, ResultFile="Results.csv"):

        """Save organoid data to a CSV file.

        Parameters
        ----------
        data : list of dict
            Organoid measurements to save, each element a dict or iterable
            of dicts mapping headers to values.

        headers : list of str
            Column names for the CSV file.

        ResultFile : str, default="Results.csv"
            CSV filename to save in ``self.path``.

        Returns
        -------
        None

        Notes
        -----
        - Existing files with the same name will be overwritten.
        - Data should match the headers for correct CSV output.
        """

        full_path = self.path + PATH_DELIMITER + ResultFile
        
        #Open csv file to store segmentation results
        f = open(full_path, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        #Save data
        for row in data:
            writer.writerows(row)

        #Close file
        f.close()

    @log_method()
    def manual_discard(self, overlay=None, newResultFile="UpdatedResults.csv", ResultFile="Results.csv", SCALE_RATIO: int=6):

        """Interactively discard organoids from overlay images and update results.

        Allows manual removal of organoids by clicking their positions in an
        overlay montage. Updates CSV file with a boolean ``Manually Discarded``
        field.

        Parameters
        ----------
        overlay : numpy.ndarray, optional
            Preloaded overlay image. If None, loads from disk.

        newResultFile : str, default="UpdatedResults.csv"
            CSV filename for updated results.

        ResultFile : str, default="Results.csv"
            Original CSV file with organoid data.

        SCALE_RATIO : int, default=6
            Rescaling factor for bounding box coordinates.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If overlay montage is not found, returns without opening the figure.

        Raises
        ------
        ValueError
            If the original CSV results file does not exist.

        Notes
        -----
        - Organoid coordinates are mapped to hive positions using
        ``get_hive_coord_montage``.
        - Interactive figure connects mouse clicks to discard/undiscard
        functions.
        - Updated CSV file flags discarded organoids with
        ``Manually Discarded``.
        - Overlay figure axes ticks are removed for clarity.
        """

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Launching Manual Discard")
        
        #Verify Overlay montage exist
        overlay_path = self.path + PATH_DELIMITER +  "Montage_" + ImageType.OVERLAY.value + self.from_parents["exp_name"] + "_" + self.from_parents["substrate_name"] + "_" + self.name + "_" + ChannelNames.BF.name + ".jpg"
        if os.path.isfile(overlay_path) == False:
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Montage Overlay not found --> skipped.")
            return
        
        #Open overlay image
        if overlay is None:
            overlay = np.array(imread(overlay_path))
        else:
            overlay = overlay

        #Read and store data
        ResultFile_path = self.path + PATH_DELIMITER + ResultFile
        if os.path.isfile(ResultFile_path) == False:
            logger.warning(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Results file not found --> skipped.")
            return
            # return ValueError(f"Result file {ResultFile_path} do not exist.")
        f = open(self.path + PATH_DELIMITER + ResultFile, 'r', newline='')
        spamreader = csv.reader(f, delimiter=',')
        data=[]
        for i,row in enumerate(spamreader):
            if i == 0: headers = row
            else:
                data.append({})
                [data[-1].update({headers[j]: row[j]}) for j in range(len(headers))]
        f.close()

        #Get index for aquisition order
        idx = uf.idx_acquisition_order(self.current_param["Aquisition order"])
        #Get hive coordinates
        coord_hives, idx_flatten = uf.get_hive_coord_montage(overlay.shape, idx, base='montage')
        #Rescale coordinates due to rescale
        coords = []
        for i in range(len(data)):
            bbox = (
                float(data[i]["bbox_ymin"])/SCALE_RATIO, 
                float(data[i]["bbox_xmin"])/SCALE_RATIO, 
                float(data[i]["bbox_ymax"])/SCALE_RATIO,  
                float(data[i]["bbox_xmax"])/SCALE_RATIO
                )
            coords.append({"Index": data[i]["Index"], \
                            "Hive number": data[i]["Image number"], \
                            "Hive coord": coord_hives[idx_flatten.index(int(data[i]["Image number"]))],
                            "bbox": bbox 
                            })
        
        #Create figure
        fig_name = f"{self.from_parents["substrate_name"]}_{self.name} Manual Discard"
        fig3, ax3 = plt.subplots(num=fig_name)
        #Remove ticks label
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.imshow(overlay)
        discarded_organoids = [] #will hold the discarded hives
        handler_on_click = partial(uf.on_click_discard_undiscard_organoid, coords=coords, discarded_organoids=discarded_organoids) #to enable passing variable to plt.connect functions
        fig3.canvas.mpl_connect('button_press_event', handler_on_click) #callbaks to discard/undiscard hives when selecting a hive on the picture
        handler_on_close = partial(uf.save_before_close, path=self.path, name=fig_name)
        fig3.canvas.mpl_connect('close_event', handler_on_close)
        plt.show()

        #Save Manual Discard image

        #Once picture is closed, update data & update results file
        f = open(self.path + PATH_DELIMITER + newResultFile, 'w+', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        if len(discarded_organoids) > 0:
            for i in range(len(data)): #loop over each picture (ie hive)
                #print(data[s][d][i], type(data[s][d][i]), len(data[s][d][i]))
                if data[i]["Index"] in discarded_organoids:
                    data[i]["Manually Discarded"] = True
                else:
                    data[i]["Manually Discarded"] = False
            writer.writerows(data)
        else: #simply copy the data in UpdatedResults.csv
            writer.writerows(data)
        f.close()
        
        logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.name} | Discarded organoids are: {discarded_organoids}")

class Hive:

    #parent: Day #enable autocompletion and will hold the calling Day instance

    def __init__(self, hive_number, image_number, from_parents, hexagon, filenames_original):

        self.hive_number = hive_number
        self.image_number = image_number
        self.from_parents = from_parents
        self.filenames_original = filenames_original
        self.dimension = {
            "site": len(filenames_original), 
            "timepoint": len(filenames_original[0]), 
            "z-stack": len(filenames_original[0][0]),
            "n_channel": len(filenames_original[0][0][0]),
            "channel": filenames_original[0][0][0]}
        self.from_parents = from_parents
        self.hexagon = hexagon
        self.name = from_parents["exp_name"] + "_" + from_parents["substrate_name"] + "_" + from_parents["day_name"] + "_" + str(self.hive_number)
        self.results = []
        self.crop_coord = None
        self.opti_crop_coord = None
        self.threshold = [[[ {chan: None for chan in self.dimension["channel"]} for _ in range(self.dimension["z-stack"]) ] \
                for _ in range(self.dimension["timepoint"])] \
                    for _ in range(self.dimension["site"])]
        self.hive_crop_mask = None
        self.opti_crop_mask = None
        
        #Update filenames
        self.filenames_crop = self.update_filename(ImageType.CROP)
        self.filenames_optimized_shadow = self.update_filename(ImageType.OPTIMIZED_SHADOW)
        self.filenames_mask = self.update_filename(ImageType.MASK)
        self.filenames_overlay = self.update_filename(ImageType.OVERLAY)

    def get_attribute_name(self, image_type: ImageType, prefixe: str, suffixe: str):

        """Return the value of a dynamically constructed attribute for the object.

        This method builds an attribute name by concatenating a prefix, the
        lowercased name of an `ImageType` enum, and a suffix. The resulting
        attribute is then retrieved from the current object using `getattr`.

        Parameters
        ----------
        image_type : ImageType
            Enum value representing the type of image. Its name is used in
            constructing the attribute name.

        prefixe : str
            String to prepend to the `image_type` name.

        suffixe : str
            String to append to the `image_type` name.

        Returns
        -------
        Any
            The value of the attribute corresponding to the constructed name.

        Raises
        ------
        AttributeError
            If the constructed attribute name does not exist in the object.

        Examples
        --------
        >>> obj.get_attribute_name(ImageType.CROP, "filenames_", "")
        [... list of crop filenames ...]
        """
        
        attr_name = f"{prefixe}{image_type.name.lower()}{suffixe}"
        return getattr(self, attr_name)

    def update_filename(self, image_type: ImageType, filenames_from_day=None):
        """Retrieve the updated filename for a specific image type for this hive.

        This method determines the index of the current hive within its parent
        day (`self.parent.filenames_original`) and returns the corresponding
        filename from the parent's attribute for the given `ImageType`.
        If the filename cannot be determined (e.g., the hive is not found or
        the attribute does not exist), it returns `None`.

        Parameters
        ----------
        image_type : ImageType
            Enum specifying the type of image (e.g., CROP, MASK, OVERLAY).
            Used to construct the parent attribute name containing filenames.

        Returns
        -------
        str or None
            The filename corresponding to this hive and image type, or `None`
            if the filename could not be retrieved.

        Notes
        -----
        - Relies on the parent object having an attribute named using the
        pattern ``filenames_{image_type.name.lower()}``.
        - This method is used to keep hive-level filenames synchronized with
        the parent day/session attributes.
        """

        try:
            filenames_type = "filenames_" + image_type.name.lower()
            #filename_attribute = self.get_attribute_name(image_type, "filenames_", "")
            for h_i, h in enumerate(self.from_parents["filenames_original"]):
                if self.filenames_original == h:
                    idx = h_i

            if filenames_from_day is None:
                return self.from_parents[filenames_type][idx]
            else:
                return filenames_from_day[idx]
        except:
            return None

    def open_image(self, image_type, slices, rgb=False):

        """Open a specific image for a hive given its type and slice indices.

        This method retrieves the filename corresponding to the selected
        `ImageType` and slice, checks that the file exists, loads it as a
        NumPy array, and optionally converts it to grayscale.

        Parameters
        ----------
        image_type : ImageType
            Type of image to open (e.g., ORIGINAL, CROP, MASK, OVERLAY).

        slices : Tuple[int, int, int, ChannelNames]
            Indices specifying (site, timepoint, z-plane, channel) of the image.

        rgb : bool, default=False
            If True, retain RGB channels; if False, convert multi-channel
            images to grayscale.

        Returns
        -------
        numpy.ndarray
            The image as a NumPy array. Grayscale images have shape (height, width),
            RGB images have shape (height, width, 3).

        Raises
        ------
        ValueError
            If the specified image file does not exist.

        Notes
        -----
        - Multi-channel images with more than 3 channels are truncated to the first
        3 channels.
        - Uses ``self.get_attribute_name`` to dynamically retrieve filenames.
        - Path construction depends on the hierarchical structure of
        parent objects (`parent.parent.parent.path` etc.).
        """
        file_name = self.get_attribute_name(image_type=image_type, prefixe="filenames_", suffixe="")
        filepath = self.from_parents["day_path"] + file_name[slices[0]][slices[1]][slices[2]][slices[3]]
        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Opening image: slice={slices}, filename={file_name[slices[0]][slices[1]][slices[2]][slices[3]]}")

        #Determine filename & check it exists
        file_name = self.get_attribute_name(image_type=image_type, prefixe="filenames_", suffixe="")
        filepath = self.from_parents["day_path"] + file_name[slices[0]][slices[1]][slices[2]][slices[3]]
        if os.path.isfile(filepath) == False:
            logger.error(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Image do not exist: {filepath}")
            raise ValueError(f"{filepath} not found. {file_name} couldn't be opened.")

        #Open image
        img = np.array(imread(filepath))

        #Re-adjust dimension if needed. SHOULDN'T BE NECESSARY !!!
        if len(img.shape) > 2:
            if img.shape[2] > 3:
                img = img[:,:,0:3]

        #RGB or Grayscale
        if rgb == False and len(img.shape) > 2:
            img = rgb2gray(img)

        return img
    
    def find_hive_rotation_angle(self, hive_mask, angle_user, angle_range, angle_step, SCALE_RATIO, slices):

        """Compute the optimal rotation angle for a hive image using convolution.

        This method evaluates the alignment of a given hive mask with the image
        by rotating the mask over a range of angles and computing the convolution
        maximum for each rotation. The angle corresponding to the highest
        convolution maximum is returned as the best rotation.

        Parameters
        ----------
        hive_mask : numpy.ndarray
            Binary mask of the hive shape to use for convolution.

        angle_user : float
            Reference angle around which the search is centered (degrees).

        angle_range : float
            Maximum deviation from `angle_user` to consider (±degrees).

        angle_step : float
            Incremental step for angle evaluation (degrees).

        SCALE_RATIO : int
            Factor to downscale the image before convolution to speed up
            computation.

        slices : Tuple[int, int, int, ChannelNames]
            Slice indices specifying (site, timepoint, z-plane, channel) for the image.

        Returns
        -------
        float
            The rotation angle (in degrees) that maximizes the convolution
            between the rotated mask and the hive image.

        Notes
        -----
        - The hive image is resized by `SCALE_RATIO` to accelerate computation.
        - Convolution is computed using FFT convolution (`fftconvolve`).
        - The method assumes `self.open_image` correctly loads and converts the
        image to grayscale if needed.
        """
        
        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Computing rotation angle")
        
        #Open image
        img = self.open_image(ImageType.ORIGINAL, slices)
        img = resize(img, (round(img.shape[0]/SCALE_RATIO), round(img.shape[1]/SCALE_RATIO)))
        
        angles = np.arange(angle_user-angle_range, angle_user+angle_range, angle_step)
        conv_max_value = [] #will hold list on maximum value of each convolution output

        for a in angles:
            conv = fftconvolve(img, rotate(hive_mask, a), mode='same') #compute convolution of hive image with a rotation of hive mask
            conv_max_value.append(conv.flatten().max())
        
        best = angles[conv_max_value.index(max(conv_max_value))]

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Rotation angle={best}degree")

        return best

    def crop_hive(self, SCALE_RATIO, best_angle, slices: Tuple[int, int, int, ChannelNames], save=True):
        """Crop and mask a single hive image using the optimal rotation angle.

        This method performs the following steps:

        1. Opens the original hive image for the specified slice.
        2. Rotates the image by the `best_angle`.
        3. Adjusts image dimensions to ensure the hexagon fits.
        4. Computes the hive barycenter (for BF channel) and generates a
        filled hexagonal mask.
        5. Crops the image around the hive using the mask.
        6. Assigns zero to pixels outside the hive and optionally saves
        the cropped image to disk.

        Parameters
        ----------
        SCALE_RATIO : int
            Factor used to downscale or adjust image dimensions during
            barycenter computation.

        best_angle : float
            Rotation angle (in degrees) to apply to the hive image.

        slices : Tuple[int, int, int, ChannelNames]
            Indices specifying the slice to process: (site, timepoint, z, channel).

        save : bool, default=True
            If True, the cropped image is saved to disk in TIFF format
            under the parent's crop directory.

        Returns
        -------
        None

        Attributes Set
        --------------
        hive_crop_mask : numpy.ndarray
            Binary mask of the hive (1 inside, 0 outside) for the cropped image.

        crop_coord : Tuple[int, int, int, int]
            Coordinates used for cropping the image: (y_min, y_max, x_min, x_max).

        Notes
        -----
        - The mask and cropping are computed only when the channel is BF.
        - Pixels outside the hexagonal hive are set to zero.
        - Saves the cropped image as TIFF, replacing `.jpg` or `.jpeg` extensions.
        - Relies on `self.open_image` and `self.get_hive_barycenter`.
        - The method preserves the image's original data type.
        """

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Croping rotation angle")

        #Open image
        img = self.open_image(ImageType.ORIGINAL, slices, rgb=True) #open image
        img_type = img.dtype

        #rotate image
        img = rotate(img, -best_angle, preserve_range=True, resize=False) #weird, before needed to write resize=True and now need o write False
        img = img.astype(img_type)
        img = uf.adjust_dimension_after_rotation(img, [round(self.hexagon["height_pxl"]), round(self.hexagon["diagonal_pxl"])], img_type) #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)

        if ChannelNames.BF.name == slices[3]:

            (y_c, x_c) = self.get_hive_barycenter(best_angle, self.hexagon["hexagon_rescale"], SCALE_RATIO, ImageType.ORIGINAL, slices) #compute hive barycenter

            #Compute x and y values where crop should occure:
            y_cut = self.cut_value_for_crop(y_c, round(self.hexagon["height_pxl"]/2), img.shape[0])
            x_cut = self.cut_value_for_crop(x_c, round(self.hexagon["diagonal_pxl"]/2), img.shape[1]) #round here to get consistent dimension over pictures

            # Fill the hexagon mask with 1 value
            hive = HexagonMath.hexagon_outline_ndarray(img.shape, (y_c, x_c), self.hexagon["side_pxl"], thickness=5, return_type='hexagon')
            hive = flood_fill(hive, (y_c, x_c), 1) #fill the hexagon
            hive = hive.astype(int)
            #Force inner part to be set to at 1
            if hive[y_c, x_c] == 0:
                hive = np.invert(hive)
            self.hive_crop_mask = hive

            #Update crop_coord attribute
            self.crop_coord = (y_cut[0], y_cut[1], x_cut[0], x_cut[1])

        #Crop the picture
        img[self.hive_crop_mask == 0] = 0 #assign all values outside hive to 0
        if len(img.shape) > 2:
            img = img[self.crop_coord[0]:self.crop_coord[1], self.crop_coord[2]:self.crop_coord[3], :] #crop around h and diag
        else:
            img = img[self.crop_coord[0]:self.crop_coord[1], self.crop_coord[2]:self.crop_coord[3]] #crop around h and diag

        #Save the cropped picture
        if save == True:
            if PATH_DELIMITER in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
                filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(PATH_DELIMITER)
                sub_folder = filename[0] + PATH_DELIMITER
                filename = filename[1]
            else:
                sub_folder = ""
                filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
            out_path = self.from_parents["day_path"] + sub_folder + ImageType.CROP.value + filename
            if '.jpg' in out_path:
                out_path = out_path.replace('.jpg', '.tif')
            elif '.jpeg' in out_path:
                out_path = out_path.replace('.jpg', '.tif')
            imsave(out_path, img)

    def optimized_crop_hive(self, hives_mask_side, k_step, slices: Tuple[int, int, int, ChannelNames], display='off'):

        """Refine the crop of a hive image using side-wise analysis of pixel intensity.

        This method performs a fine-tuned hexagonal crop of a hive image by analyzing
        pixel intensity variations along the six sides of the hive. It identifies the
        end of shadows and adjusts the vertices of the hexagon to optimize cropping.
        The resulting optimized mask is applied to the image, and the cropped image
        is optionally saved.

        Parameters
        ----------
        hives_mask_side : list of numpy.ndarray
            List of coarse masks for each slice and each side of the hexagon.
            Used to compute pixel intensity statistics for optimized cropping.

        k_step : int
            Step size (in pixels) for sampling slices along the hive side when
            computing the optimized crop.

        slices : tuple(int, int, int, ChannelNames)
            Indices specifying the slice to process: (site, timepoint, z, channel).

        display : {'on', 'off'}, default='off'
            If 'on', plots the pixel intensity statistics and derivative for
            each side of the hexagon.

        Returns
        -------
        None

        Attributes Set
        --------------
        opti_crop_mask : numpy.ndarray
            Binary mask of the finely tuned hexagon used for cropping the image.

        Notes
        -----
        - This method operates only on BF channel images for side-wise analysis.
        - Pixels outside the optimized hexagon are set to zero.
        - Saves the optimized cropped image as TIFF, replacing `.jpg` or `.jpeg`
        extensions.
        - Relies on `self.open_image` and previously computed coarse `hives_mask_side`.
        - The method preserves the image's original data type.
        - Visualization is optional and only active when `display='on'`.
        """

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Refining crop")
        
        #Open image in grayscale
        img = self.open_image(ImageType.CROP, slices, rgb=True)

        if ChannelNames.BF.name == slices[3]:
            
            #Convert to grayscale
            if len(img.shape) > 2: img_gray = rgb2gray(img)
            else: img_gray = img.copy()

            if display == 'on':
                fig, ax = plt.subplots(2,6)

            #Variables
            img_shape = img_gray.shape #shape of the picture
            y_c, x_c = uf.get_img_center(img_shape, axes=1)
            # x_c = round(img_shape[1]/2) #center of the picture along x axis
            # y_c = round(img_shape[0]/2) #center of the picture along y axis
            coord = [] #will hold the coordinates of the vertices of the fine tunned hexagon
            hive_mask = np.zeros(img_gray.shape) #will hold the mask of the fine tuned hexagon
            cut_idx = [None]*6 #will hold the for each of the 6 side of the hexagon the slice where to cut the hexagon

            #Compute the mean and (quartile 3 - quartile 1) length of the pixel values in all slices of each heagon side
            for s_k in range(6): #loop over the different side of the hexagon
                stats = []
                for k in range(len(hives_mask_side)): #loop over each slice of hexagon
                    roi = img_gray[hives_mask_side[k][s_k] > 0].flatten()
                    #Compute statistical info about the pixel of the side s_k and slice kth of 10pixel thick
                    stats.append(dict(mean=np.mean(roi)))
                    #Extract means of each slice and compute the derivative over the slices
                    means = [elt["mean"] for elt in stats]
                    if display == 'on':
                        ax[0,s_k].plot(range(0,len(means)), means)

                #Compute the means and deltaQ variations
                means_derivative = []
                for k in range(len(means)-1):
                    means_derivative.append((float(means[k+1]-means[k])/2))
                    if display == 'on':
                        ax[1,s_k].plot(range(0,len(means_derivative)), means_derivative)

                #Determine the slice at which there is no more shadow,
                #ie when there is a high change in pixel intensity,
                #ie at 80% of the length between the maximum of the derivative and the 1st following 0
                mean_derivative_max = means_derivative.index(max(means_derivative))
                mean_derivative_1st_0 = means_derivative[-1]
                for elt in range(mean_derivative_max, len(means_derivative)):
                    if means_derivative[elt] < 0:
                        mean_derivative_1st_0 = elt
                        break
                if mean_derivative_1st_0 == means_derivative[-1]:
                    mean_derivative_1st_0 = 15
                cut_idx[s_k] = round((mean_derivative_1st_0 - mean_derivative_max)*0.8 + mean_derivative_max)

                if display == "on":
                    ax[1,s_k].scatter(mean_derivative_max, 0, c='b')
                    ax[1,s_k].scatter(mean_derivative_1st_0, 0, c='g')
                    ax[1,s_k].scatter(cut_idx[s_k], 0, c='r')

                #Store the coordinates of the fine tuned vertices of the side s_k
                out = HexagonMath.hexagon_outline_ndarray(img_shape, [y_c, x_c], round((self.hexagon["height_pxl"] - 2*cut_idx[s_k]*k_step)/(2*np.sin(np.pi/3))), thickness=5, return_type='coord-side')
                out = out[2*s_k : 2*s_k+1]
                coord = coord + out
                if s_k == 5: #add the first point at the end to be able to draw the line between point 5 and point 0
                    coord.append(coord[0])

            #Have an hexagonal shape for coordinates (ie 0° angle between the bottom and top side of the hexagon)
            if coord[1][1] != coord[2][1]:
                y_min = min(coord[1][1], coord[2][1])
                coord[1][1] = y_min
                coord[2][1] = y_min
            if coord[4][1] != coord[5][1]:
                y_max = max(coord[4][1], coord[5][1])
                coord[4][1] = y_max
                coord[5][1] = y_max

            #Draw the fine tuned hive
            for s_k in range(6):
                x0, y0 = coord[s_k][0], coord[s_k][1]
                x1, y1 = coord[s_k+1][0], coord[s_k+1][1]
                rr, cc = line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
                # For each pixel in the line, draw a small disk of radius thickness//2
                for r, c in zip(rr, cc):
                    dr, dc = disk((r, c), radius=5 // 2, shape=hive_mask.shape)
                    hive_mask[dr, dc] = 1

            #Force inner part to be set to at 1
            hive_mask = flood_fill(hive_mask, (x_c, y_c), 1)
            hive_mask = hive_mask.astype(int)
            if hive_mask[y_c, x_c] == 0:
                hive_mask = np.invert(hive_mask)
            self.opti_crop_mask = hive_mask

        #Save the Optimized cropped picture
        img[self.opti_crop_mask == 0] = 0
        if PATH_DELIMITER in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(PATH_DELIMITER)
            sub_folder = filename[0] + PATH_DELIMITER
            filename = filename[1]
        else:
            sub_folder = ""
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
        out_path = self.from_parents["day_path"] + sub_folder + ImageType.OPTIMIZED_SHADOW.value + filename
        if '.jpg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        elif '.jpeg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        imsave(out_path, img)

        plt.show()

    def otsu(self, slices: Tuple[int, int, int, ChannelNames], img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Compute and store the Otsu threshold for a hive image.

        This method opens the optimized cropped image, converts it to grayscale,
        rescales pixel values to [0,1], computes the Otsu threshold using `sckimage.otsu`
        , and stores it in the hive's threshold attribute.

        Parameters
        ----------
        slices : Tuple[int, int, int, ChannelNames]
            Indices specifying the slice to process: (site, timepoint, z, channel).

        Returns
        -------
        float
            The computed Otsu threshold.

        Notes
        -----
        - Uses `ImageType.OPTIMIZED_SHADOW` images.
        - Grayscale conversion is applied if the image has multiple channels.
        - Threshold is stored in `self.threshold` for later use.
        """
        
        #Open image in grayscale
        img = self.open_image(img_type, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()

        #Turn image into range [0,1]
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

        threshold = threshold_otsu(img_gray)

        #Update threshold attribute
        self.threshold[slices[0]][slices[1]][slices[2]][slices[3]] = threshold

        return threshold

    def custom_threshold_hive(self, slices: Tuple[int, int, int, ChannelNames], fig1=None, ax1=None, title=None, verbose='off', img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Determine custom thresholds for a hive image using histogram peak analysis and Gaussian fitting.

        This method processes the optimized shadow image for a specific slice, 
        converts it to grayscale, applies Gaussian blur to remove small artifacts, 
        computes a histogram of pixel intensities, identifies two main peaks (low 
        and high intensity), fits Gaussians to these peaks, and determines a 
        custom intensity range `[pxl_min, pxl_max]` for thresholding the hive.

        Parameters
        ----------
        slices : tuple(int, int, int, ChannelNames)
            Indices specifying the slice to process: (site, timepoint, z, channel).

        fig1 : matplotlib.figure.Figure, optional
            Figure object to plot histogram and fits. If None, a new figure is created.

        ax1 : matplotlib.axes.Axes, optional
            Axes object to plot histogram and fits. If None, a new axes is created.

        title : str, optional
            Title for the plot when displaying histogram and Gaussian fits.

        verbose : str, default 'off'
            If 'on', prints detailed information about peaks, fits, and thresholds.

        Returns
        -------
        list of float
            `[pxl_min, pxl_max]` defining the intensity range for the hive. 
            Values are in the range [0,1].

        Notes
        -----
        - Uses `ImageType.OPTIMIZED_SHADOW` images.
        - Gaussian blur is applied with a radius proportional to `self.parent.um2pxl`.
        - The histogram is split around the Otsu threshold to identify low and high intensity peaks.
        - Gaussian fits to the peaks are used to define the lower and upper thresholds.
        - Thresholds are stored in `self.threshold[slices[0]][slices[1]][slices[2]][slices[3]]`.
        - If peak detection or fitting fails, default thresholds `[0, 1]` are returned.
        - When `fig1` and `ax1` are provided, the method annotates the histogram with peak positions and fitted curves.
        """

        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Computing cutom threshold")
        
        if fig1 is None and ax1 is None:
            fig, ax = plt.subplots()
        else:
            fig = fig1
            ax = ax1
        
        #Open image in grayscale
        img = self.open_image(img_type, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()
        
        #Turn image into range [0,1]
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

        #Remove artefacts
        r_blur=5*self.from_parents["um2pxl"] #to blur unique cells entierly (considered cells about 10µm diameter)
        img_gray = gaussian_filter(img_gray, r_blur)

        pxl_depth = 255

        #Get histogram with 255 bins in range 0:1
        if img_gray.flatten().min() < 0:
            data = ax.hist(img_gray.flatten(), bins=pxl_depth, color='gray')
        else:
            data = ax.hist(img_gray.flatten(), bins=pxl_depth, range=[0,1], color='gray')
        data_y_max = max(data[0])
        data[0][0] = 0 #remove  background from data to help finding peaks (ie the huge peaks with value 0)
    
        #Create the x axis for fit
        x=np.zeros(len(data[0]))
        for dx in range(len(data[0])):
            x[dx]=data[1][dx]+((data[1][dx+1]-data[1][dx])/2)

        #Split the histogram in two groups
        otsu = round(threshold_otsu(img_gray) * pxl_depth)

        #Find the peaks
        peaks = find_peaks(data[0], distance=10, height=max(data[0])*0.005)
        if len(peaks[0]) == 0 or len(peaks[0]) == 1:
            return [0, otsu], "otsu", "otsu" #[0, 1], "NA", "NA"

        #Get the peak info of 1st half of histogram
        peaks_low = np.where(peaks[0] < otsu)
        if len(peaks_low[0]) != 0:
            idx_low = np.where(peaks[1]["peak_heights"] == max(peaks[1]["peak_heights"][peaks_low]))[0][0]
            peak_low = [peaks[0][idx_low]/pxl_depth , peaks[1]["peak_heights"][idx_low]]
            ax.scatter(peak_low[0], peak_low[1], c='r', marker='x')
        else:
            return [0, otsu], "otsu", "otsu" #[0, 1], "NA", "NA"

        #Get the peak info of 2nd half of histogram
        peaks_high = np.where(peaks[0] >= otsu)
        if len(peaks_high[0]) != 0:
            idx_high = np.where(peaks[1]["peak_heights"] == max(peaks[1]["peak_heights"][peaks_high]))[0][0]
            peak_high = [peaks[0][idx_high]/pxl_depth , peaks[1]["peak_heights"][idx_high]]
            ax.scatter(peak_high[0], peak_high[1], c='g', marker='x')
        else:
            return [0, otsu], "otsu", "otsu" #[0, 1], "NA", "NA"

        if peaks[1]["peak_heights"][idx_high] > peaks[1]["peak_heights"][idx_low]:
            #First fit high peak
            guess_high = [
                peak_high[0], #center of gaussian
                peak_high[1], #amplitude
                1 - peak_high[0] #FWHM
            ]
            try:
                popt_high, pcov_high = curve_fit(uf.gauss_fit, x, data[0], p0=guess_high)
            except:
                popt_high = [0, 0, 0]
                print()
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
            fit_high = uf.gauss_fit(x, *popt_high)

            #Remove data from high peak to enable correct fit of low part
            data[0][otsu:-1] = 0

            #Fit low peak
            guess_low = [
                peak_low[0], #center of gaussian
                peak_low[1], #amplitude
                peak_low[0] #FWHM
            ]
            try:
                popt_low, pcov_low = curve_fit(uf.gauss_fit, x, data[0], p0=guess_low)
            except:
                popt_low = [0, 0, 0]
                print()
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
            fit_low = uf.gauss_fit(x, *popt_low)

        elif peaks[1]["peak_heights"][idx_high] < peaks[1]["peak_heights"][idx_low]:
            #First fit low peak
            guess_low = [
                peak_low[0], #center of gaussian
                peak_low[1], #amplitude
                peak_low[0] #FWHM
            ]
            try:
                popt_low, pcov_low = curve_fit(uf.gauss_fit, x, data[0], p0=guess_low)
            except:
                popt_low = [0, 0, 0]
                print()
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
            fit_low = uf.gauss_fit(x, *popt_low)

            #Remove data from low peak to enable correct fit of high part
            data[0][0:otsu] = 0

            #Fit high peak
            guess_high = [
                peak_high[0], #center of gaussian
                peak_high[1], #amplitude
                1 - peak_high[0] #FWHM
                ]
            try:
                popt_high, pcov_high = curve_fit(uf.gauss_fit, x, data[0], p0=guess_high)
            except:
                popt_high = [0, 0, 0]
                print()
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
            fit_high = uf.gauss_fit(x, *popt_high)

        #Set threshold for high pixel values depending on the spread of the fit ie when the gaussian encouters the center of the low intensity gaussian
        if popt_high[0] - popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] #center
            high_cut_type = "mu_high"
        elif popt_high[0] - 2*popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] - popt_high[2] #1 sigma
            high_cut_type = "mu_high - sigma"
        elif popt_high[0] - 2*popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] - 2*popt_high[2] #2 sigma
            high_cut_type = "mu_high - 2sigma"
        else:
            pxl_max = popt_high[0] - 3*popt_high[2] #3 sigma
            high_cut_type = "mu_high - 3sigma"

        #Set threshold for low pixel value
        if popt_low[0] < 0:
            pxl_min = 0
            low_cut_type = "0"
        elif popt_low[0] - popt_low[2] < 0:
            pxl_min = popt_low[0]
            low_cut_type = "mu_low"
        else:
            pxl_min = popt_low[0] - popt_low[2]
            low_cut_type = "mu_low - sigma"

        #Verify there is no aberation on threshold values
        if pxl_min < 0:
            pxl_min = 0
        if pxl_max <=0:
            #print("pxl_max < 0 --> Threshold not found.")
            pxl_min = 0
            pxl_max = 1
        if pxl_max - pxl_min <= 0:
            #print("Threshold not found.")
            pxl_min = 0
            pxl_max = 1

        x_cut = [pxl_min, pxl_max]

        #Prints
        if verbose == 'on':
            thresh_pxl_depth = [elt*pxl_depth for elt in x_cut]
            logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Determination of threshold values done:\n \
                        \tOtsu 2 groups thresh = {otsu}\n \
                        \tLow pixel intensity peak (x,y): {peak_low}; \tGaussian fit Params: {popt_low}; \tMin threshold defined as {low_cut_type}\n \
                        \tHigh pixel intensity peak (x,y): {peak_high}; \tGaussian fit Params: {popt_high}; \tMin threshold defined as {high_cut_type}\n \
                        \tThreshold value={thresh_pxl_depth}")
        else:
            thresh_pxl_depth = [elt*pxl_depth for elt in x_cut]
            logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Determination of threshold values done:\n \
                        \tOtsu 2 groups thresh = {otsu}\n \
                        \tLow pixel intensity peak (x,y): {peak_low}; \tGaussian fit Params: {popt_low}; \tMin threshold defined as {low_cut_type}\n \
                        \tHigh pixel intensity peak (x,y): {peak_high}; \tGaussian fit Params: {popt_high}; \tMin threshold defined as {high_cut_type}\n \
                        \tThreshold value={thresh_pxl_depth}")

        #Display curves & annotate
        if fig1 is not None and ax1 is not None:
            ax.set_title(title, fontsize=9)
            ax.set_ylim(0,500000)
            ax.annotate(f"{low_cut_type}", xy=(pxl_min+0.02, 400000), color='r', fontsize=8)
            ax.annotate(f"{round(x_cut[0], 2)}", xy=(pxl_min+0.02, 300000), color='r', fontsize=8)
            ax.annotate(f"{high_cut_type}", xy=(pxl_max+0.02, 200000), color='g', fontsize=8)
            ax.annotate(f"{round(x_cut[1], 2)}", xy=(pxl_max+0.02, 100000), color='g', fontsize=8)
            ax.plot(x, fit_low, c='r')
            ax.plot([pxl_min, pxl_min], [0, data_y_max], c='r', linestyle=':')
            ax.plot(x, fit_high, c='g')
            ax.plot([pxl_max, pxl_max], [0, data_y_max], c='g', linestyle=':')
        else:
            plt.close(fig)

        #Update threshold value
        self.threshold[slices[0]][slices[1]][slices[2]][slices[3]] = x_cut

        return x_cut#, low_cut_type, high_cut_type
    
    def generate_organoid_mask(self, slices: Tuple[int, int, int, ChannelNames], fig=None, ax=None, ax_idx=None, verbose='off',img_type: ImageType=ImageType.OPTIMIZED_SHADOW):

        """Generate a binary mask and overlay for organoids in a hive image.

        This method processes a specified slice of an optimized shadow image, applies
        thresholding, morphological filtering, and region-based criteria to segment 
        organoids. It generates a binary mask, saves it as an image file, and also 
        creates an overlay of the mask on the original image. Optionally, the method 
        can display the segmentation steps in a subplot.

        Parameters
        ----------
        slices : tuple(int, int, int, ChannelNames)
            Indices specifying the slice to process: (site, timepoint, z, channel).

        fig : matplotlib.figure.Figure, optional
            Figure object for displaying segmentation steps. If None, no figure is displayed.

        ax : matplotlib.axes.Axes, optional
            Axes object corresponding to `fig`. If None, no subplot is displayed.

        ax_idx : int, optional
            Index of the subplot column to display this hive. Only used if `fig` and `ax` are provided.

        verbose : str, default 'off'
            If 'on', prints information about the number of organoids found.

        Returns
        -------
        organoid_mask : np.ndarray
            Binary mask of the segmented organoids (dtype=uint8, values 0 or 255).

        Notes
        -----
        - Uses `ImageType.OPTIMIZED_SHADOW` images as input.
        - Thresholds for segmentation are read from `self.threshold[slices]`.
        - Morphological operations (erosion, area closing, dilation) are applied to refine the mask.
        - Organoids are filtered based on eccentricity (<0.88) and area (between 30,000 and 2,000,000 pixels).
        - Saves:
            - `ImageType.MASK` image of the organoid mask.
            - `ImageType.OVERLAY` image combining original image and mask in red overlay.
        - Optional display shows original, thresholded, filtered, labeled, and overlay images for debugging.
        """

        logger.info(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Generating mask: {self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]}")

        mesh_pxl = round(60*self.from_parents["um2pxl"]/2)     
        #Open image in grayscale
        img = self.open_image(img_type, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()

        #Turn image into range [0,1]
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

        #Remove artefacts
        r_blur=5*self.from_parents["um2pxl"] #to blur unique cells entierly (considered cells about 10µm diameter)
        img_blur = gaussian_filter(img_gray, r_blur)

        #Apply threshold
        if ChannelNames.BF.name == slices[3] :
            img_thresh = (img_blur > self.threshold[slices[0]][slices[1]][slices[2]][slices[3]][0]) \
                & (img_blur <= self.threshold[slices[0]][slices[1]][slices[2]][slices[3]][1])
        else:
            img_thresh = img_blur > self.threshold[slices[0]][slices[1]][slices[2]][slices[3]]

        #Close areas
        img_bin1 = binary_erosion(img_thresh, footprint=ellipse(mesh_pxl, mesh_pxl)) #to get rid of the segmented mesh
        img_bin2 = area_closing(img_bin1)
        img_bin = binary_dilation(img_bin2, footprint=ellipse(mesh_pxl, mesh_pxl))

        #Find the different regions, label it
        labels, num = label(img_bin, return_num=True, connectivity=2)
        regions = regionprops(labels, intensity_image=img_gray)

        # Find regions not satisfiying morphological criteria:
        # eccentricity <0.8
        # area in between 0.01-1.5 mm²
        labs = []
        for rg in regions:
            if rg.area > 30000:
                #print(rg.label, rg.eccentricity, rg.area_filled)
                pass
            #Find ones to remove
            if (rg.eccentricity > 0.88) or (rg.area_filled <= 10000*self.from_parents["um2pxl"]) or (rg.area_filled >= 1500000*self.from_parents["um2pxl"]):
                labs.append(rg.label)
        #Remove them
        for k in sorted(labs, reverse=True):
            del(regions[k-1])
            labels[(labels == k)] = 0

        if verbose == 'on':
            print()
            print(f"Image {self.filenames_original}: {np.unique(labels)} --> {len(np.unique(labels)) - 1} organoids found.")
            print()

        #Fill the holes in accurate regions
        tmp = labels.copy()
        for rg in regions:
            r_min, c_min, r_max, c_max = rg.bbox
            labels[r_min:r_max, c_min:c_max] = rg.image_filled

        #Binarize mask
        organoid_mask = labels.copy()
        organoid_mask[labels == 0] = False
        organoid_mask[labels > 0] = True
        # #Dilate a bit the mask to take borders
        # organoid_mask = ski.morphology.binary_dilation(organoid_mask, footprint=ski.morphology.ellipse(6,6))
        #Mask from 0 to 255 to be able to save it in jpeg format
        organoid_mask = organoid_mask.astype(np.uint8)
        organoid_mask[organoid_mask > 0] = 255

        #Save mask
        if PATH_DELIMITER in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(PATH_DELIMITER)
            sub_folder = filename[0] + PATH_DELIMITER
            filename = filename[1]
        else:
            sub_folder = ""
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
        out_path = self.from_parents["day_path"] + sub_folder + ImageType.MASK.value + filename
        if '.jpg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        elif '.jpeg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        imsave(out_path, organoid_mask)

        #Save overlay of raw picture + organoid mask (in red)
        overlay = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        if len(img.shape) > 2:
            overlay[:,:,0] = img[:,:,0] + organoid_mask*0.2 #R
            overlay[:,:,1] = img[:,:,1] #G
            overlay[:,:,2] = img[:,:,2] #B
        else:
            img_8bit = img_as_ubyte(img)
            overlay[:,:,0] = img_8bit + organoid_mask*0.2 #R
            overlay[:,:,1] = img_8bit #G
            overlay[:,:,2] = img_8bit #B
        out_path = self.from_parents["day_path"] + sub_folder + ImageType.OVERLAY.value + filename
        if '.jpg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        elif '.jpeg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        imsave(out_path, overlay)

        #Display
        if fig is None and ax is None:
            fig, ax = plt.subplots()
            plt.close()
        else:
            fig = fig
            ax = ax
            ax[0, ax_idx].set_title(self.name)
            ax[0, ax_idx].imshow(img_gray) #original image
            ax[1, ax_idx].imshow(img_thresh) #original image
            ax[2, ax_idx].imshow(img_bin) #original image
            ax[3, ax_idx].imshow(labels) #original image
            # ax[0, 0].set_title("Original")
            # ax[0, 0].imshow(img_gray) #original image
            # ax[0, 1].set_title("Blur")
            # ax[0, 1].imshow(img_blur) #blur
            # ax[0, 2].set_title("Threshold")
            # ax[0, 2].imshow(img_thresh) #thresh
            # ax[0, 3].set_title("Erosion")
            # ax[0, 3].imshow(img_bin1) #eroded
            # ax[1, 0].set_title("Dilation")
            # ax[1, 0].imshow(img_bin) #filtered
            # ax[1, 1].set_title("Filter")
            # ax[1, 1].imshow(tmp) #fill holes
            # ax[1, 2].set_title("Fill holes")
            # ax[1, 2].imshow(labels)
            # ax[1, 3].set_title("Overlay")
            # ax[1, 3].imshow(overlay)
    
    def get_organoid_data(self, organoid_count, slices: Tuple[int, int, int, ChannelNames], \
                          properties=OrganoidProperties.rg_properties, \
                            additionnal_props=OrganoidProperties.additionnal_props, \
                                metadata=OrganoidProperties.metadata.value, img_type: ImageType=ImageType.OPTIMIZED_SHADOW):
        """
        Return organoid data & metadata as a list of dictionnary.
        Each element is one organoid and each key is a propertie from skimage.regionprop, a custum computed propertie or a metadata.

        **label**: int
            organoid number in the hive. Note that these labels cannot be compared over days.
        **centroidX and centroidY**: tuple
            Centroid coordinate tuple (row, col) of organoid in the image (coordinates are in pixel)
        **centroid_localX and centroid_localY**: tuple
            Centroid coordinate tuple (row, col) of organoid in the bounding box (coordinates are in pixel)
        **bbox_ymin, bbox_xmin, bbox_ymax and bbox_xmax: int
            Bounding box of the organoid. Pixels belonging to the bounding box are in the half-open interval [bbox_ymin; bbox_ymax) and [bbox_xmin; bbox_xmax).
        **eccentricity**: float
            Eccentricity of the ellipse that has the same second moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
        **area_filled**: float
            area of the organoid with all the holes filled (in µm²)
        **perimeter**: float
            perimeter of the organoid (in µm)
        **equivalent_diameter_area**: float
            The diameter of a circle with the same area as the organoid (in µm).
        **axis_major_length**: float
            The length of the major axis of the ellipse that has the same normalized second central moments as the organoid (in µm)
        **axis_minor_length**: float
            The length of the minor axis of the ellipse that has the same normalized second central moments as the organoid (in µm)
        **intensity_mean**: float
            Average of intensity values in the organoid.
        **intensity_std**: float
            Standard deviation of intensity values in the organoid.
        **intensity_median**: float
            Median of intensity values in the organoid.
        **intensity_mad**: float
            Median Absolute Error of intensity values in the organoid.
        **equivalent_diameter_perimeter**: float
            The diameter of a circle with the same perimeter as the organoid (in µm).
        **wrinkling_index**: float
            Ratio of the equivalent_diameter_perimeter / equivalent_diameter_area
        **Index**: int
            similar to label but indexing starts at 0.
        **Day**: str
            Day at which organoid has been imaged (Day 0 beeing the cell seeding day).
        **Substrate**: str
            Name of the susbtrate in whiwh this organoid has been imaged
        **Image Name**: str
            Name of the original picture from which this data have been extracted
        **Hive Number**: int
            Hive number in which the organoid is located. Cf idx_acquisition_order() method to know more about it.
        **Manually Discarded: bool
            True if user manually discarded the organoid, False in other case.
        **Analysis Date**: datetime.datetime
        **Anavlysis Version**: str

        """
        
        logger.debug(f"{self.from_parents["exp_name"]} > {self.from_parents["substrate_name"]} > {self.from_parents["day_name"]} > {self.hive_number} | Extracting organoid data")

        BF_slice = (slices[0], slices[1], slices[2], ChannelNames.BF.name)
        filename = self.filenames_original[slices[0]][slices[1]][slices[2]][ChannelNames.BF.name]

        #Open BF mask
        mask_img = self.open_image(ImageType.MASK, BF_slice, rgb=True)
        
        #Open original BF images
        img_gray = self.open_image(img_type, BF_slice, rgb=True)

        #Open original Fluo images
        img_fluo = []
        if len(slices[3]) > 1:
            fluo_chan = slices[3].copy()
            fluo_chan.remove(ChannelNames.BF.name)
            for c in fluo_chan:
                sl = (slices[0], slices[1], slices[2], c)
                img_fluo.append(self.open_image(ImageType.OPTIMIZED_SHADOW, sl, rgb=True))

        #Extract info using region props
        labels = label(mask_img)
        pxl2um = 1/self.from_parents["um2pxl"]
        regions_gray = regionprops(labels, img_gray, extra_properties=(uf.intensity_median,uf.intensity_mad), spacing=(pxl2um, pxl2um))
        regions_fluo = []
        if len(slices[3]) > 1:
            for c_i, c in enumerate(fluo_chan):
                regions_fluo.append(regionprops(labels, img_fluo[c_i], extra_properties=(uf.intensity_median,uf.intensity_mad), spacing=(pxl2um, pxl2um)))
        data = []
        for rg_i, rg in enumerate(regions_gray):
            rg_props = uf.get_all_attribute_names(rg) + list(rg._extra_properties.keys())
            data.append({})
            #Properties from regionprops method
            [data[-1].update({head: rg[head]}) for head in properties if head in rg_props] #1D data
            #print(data[-1]['intensity_median'])
            data[-1].update({"centroidX": float(rg.centroid[0]), "centroidY": float(rg.centroid[1]), 
                             "centroid_localX": rg.centroid_local[0], "centroid_localY": rg.centroid_local[1]}) #2D-data
            data[-1].update({"bbox_ymin": float(rg.bbox[0]), "bbox_xmin": float(rg.bbox[1]), 
                             "bbox_ymax": rg.bbox[2], "bbox_xmax": rg.bbox[3]}) #4-D dtta
            #Additionnal properties
            data[-1]['equivalent_diameter_perimeter'] = OrganoidProperties.equivalent_diameter_perimeter(rg.perimeter)
            data[-1]['wrinkling_index'] = OrganoidProperties.wrinkling_index(rg.perimeter, rg.equivalent_diameter_area)
            #Fluo data
            if len(slices[3]) > 1:
                for c_i, c in enumerate(fluo_chan):
                    [data[-1].update({head+"_"+c: regions_fluo[c_i][rg_i][head]}) for head in rg_props if head in OrganoidProperties.fluo_props.value + OrganoidProperties.extra_props.value]
            #Metadata
            data[-1].update({"Index": organoid_count, "Day": self.from_parents["day_name"], "Substrate": self.from_parents["substrate_name"],
                             "Site": slices[0], "Timepoint": slices[1], "Z-Stack": slices[2],
                             "Image Name": filename, "Hive number": self.hive_number, "Image number": self.image_number,
                             "Manually Discarded": False, "Analysis Date": datetime.now(), "Analysis Version": VERSION}) #metadata
            organoid_count += 1 #incremente organoid count

        idx = uf.idx_acquisition_order("bottom-flip")
        idx_flat = [elt for col in idx for elt in col]
        
        return data, organoid_count

    def cut_value_for_crop(self, barycenter, crop_length, img_length):
        """Compute valid cropping bounds given a barycenter and desired crop size.

        This method ensures that the cropping window remains within the image
        boundaries. If the crop would exceed the image edges, the min/max
        coordinates are adjusted accordingly.

        Parameters
        ----------
        barycenter : float
            The center coordinate (y or x) of the region to crop.

        crop_length : float
            Half of the desired crop size along the corresponding axis.

        img_length : int
            Total length of the image along the axis being cropped.

        Returns
        -------
        list of int
            `[cut_min, cut_max]` coordinates defining the crop window.
        """

        if round(barycenter - crop_length) < 0:
            cut_min = 0
            cut_max = round(2*crop_length)
            #print('negative y')
        elif round(barycenter + crop_length) > img_length:
            cut_min = round(img_length - 2*crop_length)
            cut_max =img_length
            #print('y larger than picture')
        else:
            cut_min = round(barycenter - crop_length)
            cut_max = round(barycenter + crop_length)
        
        del(barycenter, crop_length, img_length)
        return [cut_min, cut_max]
    
    def get_hive_barycenter(self, best_angle, hive_mask, SCALE_RATIO, image_type: ImageType, slices):

        """Compute the barycenter of a hive within an image using convolution.

        This method opens the image, resizes it according to `SCALE_RATIO`, 
        applies Gaussian blur, thresholds using Otsu, rotates according to 
        `best_angle`, and convolves with the hive mask. The barycenter (y,x) 
        of the hive is computed from the convolution result and scaled to the 
        original image dimensions.

        Parameters
        ----------
        best_angle : float
            Rotation angle applied to the image before convolution (degrees).

        hive_mask : numpy.ndarray
            Binary mask of the hive shape used for convolution.

        SCALE_RATIO : int
            Factor to downscale the image for faster computation.

        image_type : ImageType
            Type of image to open (e.g., ORIGINAL, CROP).

        slices : Tuple[int, int, int, ChannelNames]
            Indices specifying the slice to process: (site, timepoint, z, channel).

        Returns
        -------
        tuple of int
            `(y_c, x_c)` coordinates of the hive barycenter in original image scale.
            Returns `(0,0)` if the hive is not detected (empty image).

        Notes
        -----
        - Uses FFT convolution to locate the hive in the thresholded image.
        - Barycenter computation is based on projection of the convolution result.
        - The result is scaled to match the original image dimensions.
        """

        #Open image
        img = self.open_image(image_type, slices)
        img = resize(img, (round(img.shape[0]/SCALE_RATIO), round(img.shape[1]/SCALE_RATIO)))

        if sum(sum(img)) == 0:
            return (0, 0)
        
        #Pre-process of the picture
        blur = gaussian_filter(img, 5/SCALE_RATIO) #blur pictire to smooth edges
        otsu_thresh = threshold_otsu(blur) #compute the otsu automatic threshold
        thresh = np.empty(blur.shape) 
        thresh[blur >= otsu_thresh] = 0 #value below thresh =0
        thresh[blur < otsu_thresh] = 1 #value above thresh =1
        rot = rotate(thresh, - best_angle, preserve_range=True, resize=True) #rotate the blured and thresholded picture to have it in good orientation

        #Find hive on picture
        conv = fftconvolve(rot, hive_mask, mode='same') #perform convolution of hive mask and pre-processes imaged (blurred, thesrloded and rotated)
        v_max = conv.max().max() #compute maximum pixel value on the results of convolution
        conv[conv < v_max] = 0

        #Compute barycenter of the convolution image
        y_c = sum(conv.sum(axis=1)*range(conv.shape[0]))/sum(conv.sum(axis=1)) #projection on y axis and computing barycenter of y
        x_c = sum(conv.sum(axis=0)*range(conv.shape[1]))/sum(conv.sum(axis=0)) #projection on x axis and computing barycenter of x

        #Convert the barycenter at good scale
        y_c = round(self.from_parents["img_original_dim"][0]*y_c/conv.shape[0])
        x_c = round(self.from_parents["img_original_dim"][1]*x_c/conv.shape[1])

        return (y_c, x_c) 
