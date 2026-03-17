from enum import Enum
from numpy import pi
import os

os_name = os.name
if os_name == 'posix':
    PATH_DELIMITER = '/'
elif os_name == 'nt':
    PATH_DELIMITER = '\\'

SCALE_RATIO=10

VERSION = "v0.1"

class Microscope(Enum):
    LEICA = "Leica"
    ZEISS_INCUBATOR = "Zeiss-Incubator"
    ZEISS_CONFOCAL = "Zeiss-Confocal"

class Magnification(Enum):
    X5 = "x5"
    X10 = "x10"
    X20 = "x20"
    X40 = "x40"

class Design(Enum):
    H2M100 = "H2M100"
    H2M150 = "H2M150"
    H2M150b = "H2M150b"
    H2M200 = "H2M200"
    PLATE = "Plate"

class ImageFormat(Enum):
    TIF = '.tif'
    OME = '.ome'
    CZI = '.czi'
    JPEG = '.jpeg'
    JPG = '.jpg'

    @classmethod
    def get_attr(cls):
        return [elt.value for elt in cls]

class ImageType(Enum):
    ORIGINAL = ""
    CROP = "Crop_"
    OPTIMIZED_SHADOW = "OptimizedShadow_"
    MASK = "Mask_"
    CORE_MASK = "CoreMask_"
    SHELL_MASK = "ShellMask_"
    OVERLAY = "Overlay_"
    MONTAGE = "Montage_"

    @classmethod
    def get_attr(cls):
        return [elt.value for elt in cls]
    
class ChannelNames(Enum):
    
    BF = 0
    DAPI = 1
    CFP = 2
    GFP = 3
    YFP = 4
    CY3 = 5
    MCHERRY = 6
    CY5 = 7
    CY7 = 8
    IR800 = 9

    @classmethod
    def uniformize_chan_name(cls, chan_name):
        
        uni_chan = None

        available = cls._member_names_

        if chan_name in ["BF", "bf", "Bf", "Transmission", "transmission"]:
            uni_chan = cls.BF.name

        elif chan_name in ["DAPI", "Dapi", "dapi"]:
            uni_chan = cls.DAPI.name

        elif chan_name in ["CFP", "Cpf", "cfp"]:
            uni_chan = cls.CFP.name

        elif chan_name in ["FITC", "Fitc", "fitc", "GFP", "Gfp", "gfp"]:
            uni_chan = cls.GFP.name

        elif chan_name in ["YFP", "Yfp", "yfp"]:
            uni_chan = cls.YFP.name

        elif chan_name in ["TRITC", "Tritc", "tritc", "CY3", "Cy3", "cy3"]:
            uni_chan = cls.CY3.name
        
        elif chan_name in ["mCherry", "mCHERRY", "mcherry", "MCHERRY"]:
            uni_chan = cls.MCHERRY.name

        elif chan_name in ["CY5", "Cy5", "cy5"]:
            uni_chan = cls.CY5.name

        elif chan_name in ["CY7", "Cy7", "cy7"]:
            uni_chan = cls.CY7.name

        elif chan_name in ["IR800", "Ir800", "ir800"]:
            uni_chan = cls.CY7.name

        else:
            raise(f"Channel {chan_name} is unkown. Currently {", ".join(available)} and declinations are available", ValueError)

        return uni_chan
    
class OrganoidProperties(Enum):
    props = ['label', 'centroidX', 'centroidY', 'centroid_localX', 'centroid_localY', 'bbox_ymin', 'bbox_xmin', 'bbox_ymax', 'bbox_xmax', 
         'eccentricity', 'area_filled', 'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length',
         'intensity_mean', 'intensity_std']
    extra_props = ['intensity_median', 'intensity_mad']
    additionnal_props = ['equivalent_diameter_perimeter', 'wrinkling_index']
    fluo_props = ['intensity_mean', 'intensity_std']
    metadata = ['Index', 'Day', 'Substrate', 'Site', 'Timepoint', 'Z-Stack', 'Channels', 'Image Name', 'Hive number', 'Image number', 'Manually Discarded','Analysis Date', 'Analysis Version']
    rg_properties = props + extra_props

    def equivalent_diameter_perimeter(perimeter):
        return perimeter/pi   
    
    def wrinkling_index(equivalent_diameter_perimeter, equivalent_diameter_area):     
        return equivalent_diameter_perimeter/equivalent_diameter_area
    
class PlotType(Enum):
    VIOLIN = "violin"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    SCATTER_MEAN = "scatter_mean"
    SCATTER_MEDIAN = "scatter_median"
    LINE_MEAN = "line_mean"
    LINE_MEDIAN = "line_median"
    SWARMPLOT = "swarmplot"

