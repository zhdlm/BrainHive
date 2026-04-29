from typing import List
from src.options import Design, ImageType, ImageFormat, ChannelNames, PlotType, PATH_DELIMITER
import os
import re
from pprint import pprint
import itertools
from scipy.stats import median_abs_deviation, shapiro, levene, kruskal, mannwhitneyu, false_discovery_control, fisher_exact, chisquare, ttest_rel, ttest_ind, wilcoxon, bartlett, f_oneway , tukey_hsd, friedmanchisquare, zscore, skew
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
from math import floor, sqrt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils.loggin_utils import log_method
import json
import logging
from string import capwords
logger = logging.getLogger(__name__)

def get_all_attribute_names(obj):
    names = set(vars(obj).keys())  # instance attributes

    # Add property names
    for name, value in obj.__class__.__dict__.items():
        if isinstance(value, property):
            names.add(name)

    return sorted(names)

def simple_warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"

#################################################################################
# OPERATION ON LIST
#################################################################################

def unique_list(non_unique_list: List[str]|List[int]):

    """Get unique value of a 1D list and keep the order of appearence of each unique value"""

    unique_list = []
    for elt in non_unique_list:
        if elt not in unique_list:
            unique_list.append(elt)

    return unique_list

def flatten_list_recursive(variable_list, final='dict'):
    """Flatten a unlimited nested list using recusrive methode"""

    out_list = []

    for i,elt in enumerate(variable_list):
        if isinstance(elt, list):
            elt_out = flatten_list_recursive(elt, final=final)
        else:
            if final == "dict":
                elt_out = elt.values()
            elif final == "str":
                elt_out = elt
            elif final == "int":
                elt_out = [elt]
            elif final == "tuple":
                elt_out = [tuple(elt)]
        
        out_list += elt_out
        
    return out_list

##################################################################################
# HIVE NUMBER & HIVE IMAGE NUMBER RELATED
##################################################################################

def image_number_from_filename(filename, format='int', pattern="00[0-1][0-9]"):
    """Returns the number written in the image as int by default or in string if format=='str'."""
        
    number = re.search(pattern, filename)
    if number is None:
        return None
    
    number = number.group()
    if format == 'int':
        return int(number)
    elif format == 'str':
        return str(number)
    else:
        logger.error(f"format unexcpected: either 'int' or 'str'.")
        raise ValueError(f"format unexcpected: either 'int' or 'str'.")
    
def idx_acquisition_order(acquisition_order):
    """Returns nested list of image number listed in static orientation.
    
    This list enables static image reconstrcution whatever `aquisition_order`
    has been used during image capture.
    - 1st column: 3 hives
    - 2nd column: 4 hives
    - 3rd column: 5 hives
    - 4th column: 3 hives
    - 5th column: 4 hives
    ```
    1   2   3   4    5

            0
        0       0
    0       1       0
        1       1
    1       2       1
        2       2
    2       3       2
        3       3
            4
    ```
    
    Parameters
    ----------
    acquisition_order: str
        Name of the aquisition order. Accesible values are:
        {'0', '2', '11', '18', '16', '7'} and their flip version
        (ex: '0-flip').

    Returns
    -------
    List of list
        Each element of the list is a list and corresponds to a column
        of the substrate. Each column list contains the image number.

    Example
    -------
    The 1st image captured corresponds to the 11th image of the picture
    captured at day 0 and substrate is not flipped. Therefore,
    aquisition_order = '11'.
    The 1st image captire corresponds to the 2nd image of the picture
    captured at day 0 and the substrate is flipped. Therefore, 
    aquisition_order = '2-flip'.
    """
    
    idx = None

    match acquisition_order:
        case 'bottom' | '0':
            idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]] #old version
        case 'bottom-flip' | '0-flip':
            idx = [[16, 15, 7], [17, 14, 8, 6], [18, 13, 9, 5, 0], [12, 10, 4, 1], [11, 3, 2]] #old version
        case 'exp001':
            idx = [[7,6,5], [15,8,4,0], [16,14,9,3,1], [17,13,10,2], [18,12,11]] #old version
        case '2' | 'bottom right':
            idx = [[2,1,0], [3,4,5,6], [11,13,9,8,7], [12,13,14,15], [18,17,10]]
        case '2-flip':
            idx = [[7,6,0], [15,8,5,1], [16,14,9,4,2], [17,13,10,3], [18,12,11]]
        case '11':
            idx = [[0,6,7], [1,5,8,15], [2,4,9,14,16], [3,10,13,17], [11,12,18]]
        case '11-flip':
            idx = [[0,1,2], [6,5,4,3], [7,8,9,10,11], [15,14,13,12], [16,17,18]]
        case '18':
            idx = [[7,15,16], [6,8,14,10], [0,5,9,13,18], [1,4,10,12], [2,3,11]]
        case '18-flip':
            idx = [[2,3,11], [1,4,10,12], [0,5,9,14,18], [6,8,15,16], [7,15,16]]
        case '16':
            idx = [[16,17,18], [15,14,13,12], [7,8,9,10,11], [6,5,4,3], [0,1,2]]
        case '16-flip':
            idx = [[11,12,18], [3,10,12,17], [2,4,9,14,16], [1,5,8,15], [0,6,7]]
        case '7':
            idx = [[18,12,11], [17,13,10,3], [16,14,9,4,0], [15,8,5,1], [7,6,0]]
        case '7-flip':
            idx = [[18,17,16], [12,13,14,15], [11,10,9,8,7], [3,4,5,6], [2,1,0]]
        case 'row':
            idx = [[a+str(b) for b in range(1,12)] for a in ["A","B","C","D","E","F","G","H"]]

    return idx

def get_hive_number(image_number, aquisition_order):
    """Static number of hive taking into consideration acquisition parameters. 1st hive is the one at the bottom, then count in snake by starting on the left."""
    
    idx = idx_acquisition_order(aquisition_order)
    idx_flat = [elt for col in idx for elt in col]
    idx_static = [11, 3, 2, 12, 10, 4, 1, 18, 13, 9, 5, 0, 17, 14, 8, 6, 16, 15, 7]
    out= None
    if isinstance(image_number, int):  
        return idx_static[idx_flat.index(image_number)]
    elif isinstance(image_number, list):
        out = []
        for i in image_number:
            out.append(idx_static[idx_flat.index(i)])
        return out
    else:
        return out

#################################################################################
# CALIBRATION
#################################################################################

def size_calibration_vplate(um2pxl_measured): #TO CODE
    
    return um2pxl_measured

#################################################################################
# OPERATION ON IMAGES
#################################################################################

def get_img_center(img_dim, axes=1):
    """Return the (y,x) coordinates of image center."""
    if axes == 1:
        return (round(img_dim[0]/2), round(img_dim[1]/2))
    elif axes == 0:
        return (round(img_dim[1]/2), round(img_dim[0]/2))

def assign_dim(design, microscope, magnification, dimension):

    if design in [Design.H2M100.value, Design.H2M150.value, Design.H2M150b.value, Design.H2M200.value]:
        height = 2000
        n_expect = 19
    elif design == Design.PLATE.value:
        height = None
        n_expect = 96

    match microscope:

        case "Leica":
            match magnification:
                case "x5":
                    if dimension == "1944x2592":
                        um2pixel = 0.9 # 1 µm corresponds to 0.9 pxl
                        pxl2um = 1.1 # 1 pxl corresponds to 1.7714 µm
                    elif dimension == "1200x1600":
                        um2pixel = 0.54 # 1 µm corresponds to 0.5645 pxl
                        pxl2um = 1.85 # 1 pxl corresponds to 1.7714 µm
                    elif dimension == "768x1024":
                        um2pixel = 0.36
                        pxl2um = 2.78
                    if desgin == [Design.PLATEV.value]:
                        um2pixel = size_calibration_vplate(um2pixel)
                case "x10":
                    if dimension == "1944x2592":
                        um2pixel = 1.8 # 1 µm corresponds to 0.9 pxl
                        pxl2um = 0.55 # 1 pxl corresponds to 1.7714 µm


        case "Zeiss-Incubator":
            match magnification:
                case "x5":
                    if dimension == "2048x2048":
                        um2pixel = 0.675
                        pxl2um = 1.48
    
    return height, n_expect, um2pixel

def add_empty_img(images, file_name, filename_pattern, n_expect):
    
    out_images = images.copy()
    out_file_name = file_name
    
    d = floor(n_expect / 10) + 1
    u = 10

    if len(out_file_name) < n_expect:
        img_number = []

        #Search picture number that exist
        for f in out_file_name:
            img_number.append(re.search('00[0-9][0-9]', f).group())

        #Determine the missing pictures by finding the missing numbers
        numbers = []
        for i in range(d):
            for j in range(u):
                n = int(str(i) + str(j))
                if n < n_expect: numbers.append("%04d" % n)
        missing_img = [x for x in numbers if x not in img_number]
        missing_idx = [int(x) for x in missing_img]
        #print(missing_img, missing_idx)

        #Add empty image and file name at good location for missing pictures
        for i in missing_idx:
            out_images.insert(i, np.zeros(images[0].shape, dtype=images[0].dtype))
            out_file_name.insert(i, filename_pattern + "%04d" % i + '.jpeg')
    
    del(images, filename_pattern, n_expect)
    return out_images, out_file_name

def get_hive_coord_montage(sz, img_idx, base=None):

    """Returns list of hive coordinates stored as tuple ((xmin, xmax), (ymin, ymax)) and list of associated hive number"""

    # Substrate is definied as follow:
    # 1st column: 3 hives
    # 2nd column: 4 hives
    # 3rd column: 5 hives
    # 4th column: 3 hives
    # 5th column: 4 hives

    if base == 'hive':
        img_sz = sz
    elif base == 'montage':
        img_sz = (round(sz[0]/5), round(sz[1]/5))
    else:
        print("No base given to get hive coordinate in montage: hive or montage. Script aborted")
        return

    coord = []

    img_idx_flat = [item for sublist in img_idx for item in sublist]

    i=0 #counting pictures in following nested loops
    r=0 #row on the substrate
    for c in [0,1,2,1,0]:
        #Initiate location of the first picture of the row
        y_min = round((5*img_sz[0] - (c+3)*img_sz[0])/2)
        y_max = round(y_min + img_sz[0])
        x_min = img_sz[1]*(r)
        x_max = x_min + img_sz[1]

        for elt in range(len(img_idx[r])):
            coord.append(((x_min, x_max), (y_min, y_max)))
            #incremente
            i += 1
            y_min += img_sz[0]
            y_max = y_min + img_sz[0]
        r+=1
    return coord, img_idx_flat

def adjust_dimension_after_rotation(image, hive_dimension, image_dtype):
    """
    returns adujsted image after rotation
    
    Parameters: 
        image (ndarray): picture after rotation to be re-adjusted
        hive_dimension (list) of two (int) elements: 1st element is the distance (in pixel) between two opposite sides of the hexagon. 2nd element is the diagonal (in pixel) of the hexagon.
        image_type (string): dtype of the original image

    Returns:
        image (ndarray): image readjusted
    """
    for i in [0,1]:

        image_out = image.copy()

        if image.shape[i] - hive_dimension[i] < 0:
            # print('height of picture is to short to fit hexagon height')
            w = hive_dimension[i] - image.shape[i] + 0.1 #addition of 0.1 to be sure 
            tmp = np.empty((hive_dimension[i], image.shape[i], 3), dtype=image_dtype)
            tmp[round(w/2):round(image.shape[i]+w/2), :, :] = image
            image_out = tmp
    
    del(image, hive_dimension, image_dtype)
    return image_out

##################################################################################
# FILENAMES RELATED
##################################################################################

def get_filenames_leica(filename_pattern, path, exp, substrate, day, image_type):
    """Leica images have only one channel (BF), no z-stack, no timelaps only one image (region) per hive"""

    #Initiate output
    filenames = []
    image_number = [] #only for hive level
    
    #Adjust filename pattern:
    params = {"exp": exp, "substrate": substrate, "day": day}
    pattern = adjust_filename_pattern_leica(filename_pattern, image_type, params)
    if "Well" in pattern:
        number_pattern = "[A-Z][0-9]*"
        pattern = pattern.split("Well")
    else:
        number_pattern = "[0-9][0-9][0-9][0-9]"
        pattern = [pattern, ""]

    other_pattern = ImageType.get_attr()
    if image_type == "":
        del other_pattern[0]
    else:
        del other_pattern[0]
        other_pattern.remove(image_type)

    #Search for files
    folder_path = path 
    if os.path.isdir(folder_path) == False:
        #warnings.warn(f"{folder_path} do not exist.", UserWarning)
        logger.warning(f"{exp} > {substrate} > {day} > | {image_type}. Path do not exist: {folder_path}")
        
    else:
        files = os.listdir(folder_path)
        files.sort()

        for f in files:
            if (all(pat in f for pat in pattern if len(pattern) > 1)) and (any(ext in f for ext in ImageFormat.get_attr()) and any(ty in f for ty in other_pattern)==False):
                if pattern[1] != "":
                    number = image_number_from_filename(f, format='str', pattern=number_pattern)
                else:
                    number = image_number_from_filename(f, format='int', pattern=number_pattern)
                if number is not None:
                    filenames.append([[[{ChannelNames.BF.name: f}]]])
                    image_number.append(number)
                elif image_type == ImageType.ORIGINAL.value:
                    logger.error(f"{exp} > {substrate} > {day} | Number pattern not found for image {f}\n\tNumber should be compposed of 4 integers with leading 0. Example:\n\tFor image 4: 0004. For image 11: 0011.")
                    raise ValueError(f"{exp} > {substrate} > {day} | Number pattern not found for image {f}\n\tNumber should be compposed of 4 integers with leading 0. Example:\n\tFor image 4: 0004. For image 11: 0011.")
    
    return filenames, image_number

def get_filenames_zeiss_incub_pos(path, image_type):

    filenames = []
    image_number = []
    
    #folder_path = path + PATH_DELIMITER + substrate + PATH_DELIMITER + day
    folder_path = path

    pattern = "img_channel00[0-9]_position[0-9][0-9][0-9]_time[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_z[0-9][0-9][0-9].tif"
    
    if os.path.isdir(folder_path) == False:
        logger.warning(f"Path do not exist: {folder_path}")

    other_pattern = ImageType.get_attr()
    if image_type == "":
        del other_pattern[0]
    else:
        del other_pattern[0]
        other_pattern.remove(image_type)
    no_original_pattern = ImageType.get_attr()
    del no_original_pattern[0]

    sub_folders = os.listdir(folder_path)
    sub_folders = sorted(sub_folders, key=natural_key)

    s = 0
    previous_hive = 0
    for folder in sub_folders:

        if any(elt in folder for elt in ['.jpg', '.jpeg', '.tif', '.json', 'csv']):
            continue
        
        files = os.listdir(folder_path + PATH_DELIMITER + folder)
        if len(files) > 0:

            metadata_file = folder_path + PATH_DELIMITER + folder + PATH_DELIMITER + "metadata.txt"

            #Verify existence
            if os.path.isfile(metadata_file) == False:
                logger.warning(f"Metadata file not found: {metadata_file}")
                return filenames, image_number

            #Determine image dimension (number of timepoints, z-stacks, channel)
            metadata = json.load(open(metadata_file))
            chans = metadata["Summary"]["ChNames"]
            chans_uniform = [ChannelNames.uniformize_chan_name(elt) for elt in chans]
            timepoints = metadata["Summary"]["IntendedDimensions"]["time"]
            zstacks = metadata["Summary"]["IntendedDimensions"]["z"]
            for f in files:
                if any(pat in f for pat in no_original_pattern) == False: 
                    name = "Coords-" + folder + "/" + f
                    break
            position = metadata[name]["PositionIndex"]
            all_sites = [ elt["Label"] for elt in metadata["Summary"]["StagePositions"] ]
            all_sites_position_only = [elt.split("-Site")[0] for elt in all_sites]
            current_hive = folder.split("-Site")[0]
            site_size = 0
            for elt in all_sites_position_only:
                    if current_hive == elt:
                        site_size+=1

            #Create hive filename with good dimension: filenames[hive][timeppint][z-stack]{Channel1: imagename, Channel2: imagename}
            if any(image_type in f for f in files):
                filenames.append([[[ {} for _ in range(zstacks) ] for _ in range(timepoints)] for _ in range(site_size)])
            else:
                return filenames, image_number

            for t in range(timepoints):
                for z in range(zstacks):
                    for c_i, c in enumerate(chans_uniform):

                        params = {"t": t, "z": z, "c": c_i, "position": position}
                        pattern_tmp = adjust_filename_pattern_zeiss_incub_pos(pattern, image_type, params)

                        for f in files:
                            re_out = re.fullmatch(pattern_tmp, f)
                            if (re_out != None) and (any(ext in f for ext in ImageFormat.get_attr()) and any(ty in f for ty in other_pattern)==False):
                                filenames[-1][s][t][z][c] = folder + PATH_DELIMITER + f
                                if position not in image_number:
                                    image_number.append(position)
                                break
            if previous_hive != current_hive:
                s = 0
            else:
                s += 1
            previous_hive = current_hive

    return filenames, image_number

def get_filenames_zeiss_incub_ome(path, exp, substrate, day, image_type):
    
    filenames = []
    
    #folder_path = path + PATH_DELIMITER + substrate + PATH_DELIMITER + day
    folder_path = path

    return filenames

def adjust_filename_pattern_leica(pattern, image_type, params):

    exp = params["exp"]
    substrate = params["substrate"]
    day = params["day"]
    
    updated_pattern = pattern
    updated_pattern = updated_pattern.replace("ExpName", exp)
    updated_pattern = updated_pattern.replace("Substrate", substrate)
    updated_pattern = updated_pattern.replace("Day", day)
    updated_pattern = image_type + updated_pattern

    return updated_pattern

def adjust_filename_pattern_zeiss_incub_pos(pattern, image_type, params):

    pattern = "img_channel00[0-9]_position[0-9][0-9][0-9]_time[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_z[0-9][0-9][0-9].tif"
    
    t = params["t"]
    z = params["z"]
    c = params["c"]
    position = params["position"]
    
    updated_pattern = pattern.replace("time[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]", f"time{str(t).zfill(9)}")
    updated_pattern = updated_pattern.replace("z[0-9][0-9][0-9]", f"z{str(z).zfill(3)}")
    updated_pattern = updated_pattern.replace("channel00[0-9]", f"channel{str(c).zfill(3)}")
    updated_pattern = updated_pattern.replace("position[0-9][0-9][0-9]", f"position{str(position).zfill(3)}")
    updated_pattern = image_type + updated_pattern

    return updated_pattern

##################################################################################
# PLOTS RELATED
##################################################################################

def get_compact_grid(n_subplots):

    if isinstance(n_subplots, int):
        n_sqrt = round(sqrt(n_subplots))
        if n_subplots % n_sqrt == 0:
            return n_sqrt, n_subplots // n_sqrt
        elif n_sqrt*n_sqrt > n_subplots:
            return round(n_sqrt), round(n_sqrt)
        else:
            return round(n_sqrt), round(n_sqrt) + 1
        
    elif isinstance(n_subplots, tuple):
        if n_subplots % n_sqrt == 0:
            return n_sqrt, n_subplots // n_sqrt
        elif n_sqrt*n_sqrt > n_subplots:
            return round(n_sqrt), round(n_sqrt)
        else:
            return round(n_sqrt), round(n_sqrt) + 1

def on_click_discard_undiscard_organoid(event, coords=None, discarded_organoids=None):
    """
    On click, discard/undiscard organoid by 1) updating the discarded_organoid output list and 2) visually by adding blue cross on the discarded organoid.

    Parameters
    ----------
    event:
    coords: list of dictionnary. Each element of the list is a dictionnary containing information about one organoid. The keys are the following:
        -Index: int. Index of the organoid (ie, a label) on the montage overlay picture
        -Hive number: int. Number of the hive containing the organoid on the montage overlay picture.
        -Hive coord: tuple. Coordinates of the hive containing the organoid on the montage overlay picture. Written as ((xmin, xmax), (ymin, ymax))
        -bbox: tuple. Coordinates of the organoid in the picture containing only the hive. Written as (ymin, xmin, ymax, xmax)
    discarded_organoids: list of int. Contains the index of eahc organoid that have manually been discarded. Initially the list is empty and it it filled during this function upon click.
    """
    
    if event.button is MouseButton.LEFT and event.inaxes: #Do smt only if click inside plot area

        #print(f'Click at location {event.xdata}, {event.ydata}, {type(event.xdata)}')

        for i,elt in enumerate(coords): #loop over the organoid list

            if elt["Hive coord"][0][0] <= event.xdata < elt["Hive coord"][0][1] and elt["Hive coord"][1][0] <= event.ydata < elt["Hive coord"][1][1]: #Check if click is in hive

                #Generate a sub_list of organoids of this specific hive
                sub_list =  []
                [sub_list.append(elt) for j in range(len(coords)) if coords[j]["Hive number"] == elt["Hive number"]]
                #print(f"Organoids inside hives: {sub_list}")

                for sub_elt in sub_list:

                    #Coordinates of the organoid in the montage image
                    xmin = round(sub_elt["bbox"][1] + elt["Hive coord"][0][0])
                    xmax = round(sub_elt["bbox"][3] + elt["Hive coord"][0][0])
                    ymin = round(sub_elt["bbox"][0] + elt["Hive coord"][1][0])
                    ymax = round(sub_elt["bbox"][2] + elt["Hive coord"][1][0])

                    if  xmin <= event.xdata < xmax and ymin <= event.ydata < ymax: # Check if click is on organoid
                        #print(f"Selected this organoid: {sub_elt}")

                        if len(discarded_organoids) == 0 or sub_elt["Index"] not in discarded_organoids: # Check if organoid has already been discarded
                            discarded_organoids.append(sub_elt["Index"]) #add index to the discarded list
                            event.inaxes.plot([xmin, xmax], [ymax, ymin], color='c') #Add the 1st line to form the cross
                            event.inaxes.plot([xmax, xmin], [ymax, ymin], color='c') #Add the 2nd line to form the cross
                        
                        elif sub_elt["Index"] in discarded_organoids: #If already in list remove it
                            discarded_organoids.remove(sub_elt["Index"])
                            for ln in event.inaxes.lines: #loop over all the lines created in the figure
                                bbox = ln.get_bbox() #get the bounding box of each line

                                if bbox.x0 == xmin and bbox.x1 == xmax and bbox.y0 == ymin and bbox.y1 == ymax: #Check which lines have good coordinates
                                    ln.remove() #remove the line
                        #print(f"Updated list of driscarded organoids: {discarded_organoids}")
                        
                        break #to exit loop if organoid is found

        
        event.canvas.draw_idle() #Update plot

def save_before_close(event, path=None, name=None):
    fig = event.canvas.figure
    complete_path = path + PATH_DELIMITER + name + '.jpg'
    fig.savefig(complete_path, dpi=300)

def auto_bins(y):

    """Computes optimal number of bins, bin edges and counts on each bin for array y."""
    
    #Sturge bin
    n = len(y)
    sturges_bins = int(np.ceil(np.log2(n) + 1))

    #Freedman-Diaconis bin
    q75, q25 = np.percentile(y, [75, 25])
    iqr = q75 - q25
    fd_bin_width = 2 * iqr / n**(1/3)
    if fd_bin_width != 0:
        fd_bins = int(np.ceil((y.max() - y.min()) / fd_bin_width))
    else:
        fd_bins = 0

    #Select the bigger amount of bins:
    n_bins = max(sturges_bins, fd_bins)

    #Compute bin edges
    bin_edges = np.linspace(y.min(), y.max(), n_bins + 1)

    #Compute counts for each bin
    bin_count = []
    for i in range(len(bin_edges)-1):
        if i < len(bin_edges)-2:
            counts = list((y >= bin_edges[i]) & (y < bin_edges[i+1]))
        else:
            counts = list(y >= bin_edges[i])

        bin_count.append(counts.count(True))

    return n_bins, bin_edges, bin_count

def swarm_scatter_vals(y, z, spacing):

    """Returns the sorted x, y positions to create swarmplot (not optimal rendering) and associated z grouping values."""

    #Transform y into array and sort it
    y = np.asarray(y)
    order = np.argsort(y)
    y = y[order]
    z = z[order]
    #Initiate x output
    x = np.zeros(len(y))

    #Compute best bin edges and counts
    _, edges, counts = auto_bins(y)

    for bin in range(len(edges)-1):

        #Pass to next bin if there is no count
        if counts[bin] == 0:
            continue

        n = 0 # iterate each time a data point is in bin
        if (counts[bin] % 2 == 0):
            nn = 0.5 # occurence on left or right side
        else:
            nn = 0

        
        for i, yi in enumerate(y):

            #Test to determine if the yi value is in current bin
            if bin < len(edges)-2:
                test = (yi >= edges[bin]) and (yi < edges[bin+1])
            else:
                test = (yi >= edges[bin])

            if test:

                #For bins composed of only one data point
                if counts[bin] == 1:
                    xi=0

                #For bins composed of odd number of data points
                elif (counts[bin] % 2 != 0):
                    #Left side
                    if n < (counts[bin] - 1)/2:
                        factor = -1
                        nn += 1
                    #Center
                    elif n == (counts[bin] - 1)/2:
                        factor = 0
                        nn = 0
                    #Right side
                    elif n > (counts[bin] - 1)/2:
                        factor = 1
                        nn +=1
                    #Compute position
                    xi = factor*spacing*nn
                elif (counts[bin] % 2 == 0):
                    #Left side
                    if n == counts[bin]/2 - 1:
                        factor = -1
                        nn = 0.5
                    elif n < counts[bin]/2:
                        factor = -1
                        nn += 1
                    #1st data point on right side
                    elif n == counts[bin]/2:
                        factor = 1
                        nn = 0.5
                    #Right side
                    elif n > counts[bin]/2:
                        factor = 1
                        nn += 1
                    #Compute position
                    xi = factor*spacing*nn
                n+=1 #incremente number of datapoint in bin
                x[i] = xi #assign x position in output variable

    return x, y, z

def swarm_z_on_conditions(ax, data, x, y, conditions, z, markersize=40, width=0.8):

    """Plots on each (x-condition) group the y data points arranged as swmarlike colored by z group."""

    # Names of grouping variable
    x_names = list(data[x].unique())
    if conditions is not None:
        condition_names = list(data[conditions].unique())
    else:
        condition_names = [""]
    z_names = sorted(data[z].unique())
    #Dimension on plot
    box_width = width / len(condition_names)
    offsets = np.linspace(-width/2 + box_width/2, width/2 - box_width/2, len(condition_names))
    spacing = np.sqrt(markersize)/600
    #Colors
    colors = sns.color_palette("tab10", len(z_names))
    z_palette = dict(zip(z_names, colors))

    for i, day in enumerate(x_names):
        for j, cond in enumerate(condition_names):

            #Subset
            if conditions is None:
                sub = data[(data[x] == day)]
            else:
                sub = data[(data[x] == day) & (data[conditions] == cond)]

            if sub.empty:
                continue

            x_center = i + offsets[j]

            x_val, y_val, z_val = swarm_scatter_vals(np.asarray(sub[y]), np.asarray(sub[z]), spacing)
            x_val = x_val + x_center


            for k in z_names:
                xk = x_val[z_val == k]
                yk = y_val[z_val == k]
                
                ax.scatter(xk, yk, c=z_palette[k], s=markersize, edgecolor="black",
                    linewidth=0.3, zorder=3)
    
    #Scatter handlers
    z_handles = [Line2D([0], [0], marker='o', color='w', label=exp,
                  markerfacecolor=col, markersize=8, markeredgecolor='black') 
           for exp, col in z_palette.items()]
    
    #Boxplot handlers
    cond_palette = sns.color_palette("deep", len(condition_names))
    cond_handles = [ Patch(facecolor="none", edgecolor=col, label=f"{cond}", linewidth=1.5)
                    for cond, col in zip(condition_names, cond_palette)]
    
    #Legend
    lgd = ax.legend(handles=cond_handles+z_handles, title="Legend", fontsize=15, title_fontsize=15)
    lgd.set_draggable(True) # to move it on figure

def plots(data: pd.DataFrame, x: str, y: str, conditions: str, plot_type: PlotType, z: str=None, savefig: str=None, scale='linear'):
    """
    Plots of y variable over time.

    Parameters
    ----------
    data: pandas DataFrame.
        Containing all column listed in "x", "y", "conditions" and "z".
    conditions: list of string.
        The list 
    y: list of PlotType.
        Containing the variable to plot on the Y-axis. The variable need to exist as column headers in the data pd.DataFrame
    plot_type: list of string.
        Containing the name of the desired plots
    """
    
    #Verify the required headers are present in data
    for elt in (x, y, conditions, z):
        check_header(elt, data.columns.values.tolist())

    # if x == "Day":
    #     x = "Day Number"

    #Create the figure and set the design
    if conditions is not None:
        condition_name = data[conditions].unique()
    else:
        condition_name = ""
    if z != None:
        z_name = data[z].unique()
    else:
        z_name="None"
    ttl = f"{plot_type.value}_x{x}_y{y}_z{z}_c{conditions}"
    fig, ax = plt.subplots(figsize=(15,10), num=ttl)
    if x is not None:
        ax.set_xlabel(set_axis_label(x), fontsize=15)
    ax.set_ylabel(set_axis_label(y), fontsize=15)
    ax.set_title(ttl.replace("_", " "))
    for ticks in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        ticks.set_fontsize(15)

    if scale == "log":
        ax.set_yscale("log")

    #Plots
    match plot_type:
        case PlotType.VIOLIN:
            sns.violinplot(data=data, x=x, y=y, hue=conditions, fill=False, ax=ax)
            add_group_counts(ax, data=data, x="Day", hue=conditions)
            if z != None:
                swarm_z_on_conditions(ax, data, x, y, conditions, z)

        case PlotType.BOXPLOT:
            if z != None:
                sns.boxplot(data=data, x=x, y=y, hue=conditions, fliersize=0, dodge=True, fill=False, ax=ax)
                swarm_z_on_conditions(ax, data, x, y, conditions, z)
                
            else:
                sns.boxplot(data=data, x=x, y=y, hue=conditions, fill=True)
            add_group_counts(ax, data=data, x=x, hue=conditions)

        case PlotType.SCATTER:
            if x == "Day":
                tmp = data[x].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                data["Day Number"] = tmp
                data["Day Number"] = data["Day Number"].astype(int) 
                x = "Day Number"
            sns.scatterplot(data=data, x=x, y=y, hue=conditions, size=z, ax=ax)

        case PlotType.SCATTER_MEAN:
            if x == "Day":
                tmp = data[x].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                data["Day Number"] = tmp
                data["Day Number"] = data["Day Number"].astype(int) 
                x = "Day Number"
            if z != None:
                means_stds = data.groupby([conditions, x], as_index=False).agg(y_mean=(y, "mean"), y_std=(y, "std"), z_mean=(z, "mean"))
                sns.scatterplot(data=means_stds, x=x, y="y_mean", hue=conditions, size="z_mean")
                print(means_stds)
            else:
                means_stds = data.groupby([conditions, x], as_index=False).agg(y_mean=(y, "mean"), y_std=(y, "std"))
                sns.scatterplot(data=means_stds, x=x, y="y_mean", hue=conditions)
            ax.errorbar(means_stds[x], means_stds["y_mean"], yerr=means_stds["y_std"],
                fmt="none",  # don't draw additional markers
                ecolor="gray",
                capsize=4
            )

        case PlotType.SCATTER_MEDIAN:
            if x == "Day":
                tmp = data[x].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                data["Day Number"] = tmp
                data["Day Number"] = data["Day Number"].astype(int) 
                x = "Day Number"
            medians_stds = (
                data.groupby([conditions, x], as_index=False)
                .agg(y_median=(y, "median"), y_mad=(y, lambda x: median_abs_deviation(x, scale='normal'))))
            sns.scatterplot(data=medians_stds, x=x, y="y_median", hue=conditions, size=z)
            plt.errorbar(medians_stds[x], medians_stds["y_median"], yerr=medians_stds["y_mad"],
                fmt="none",  # don't draw additional markers
                ecolor="gray",
                capsize=4
            )
            
        case PlotType.LINE_MEAN:
            if x == "Day":
                x = "Day Number"
            sns.lineplot(data=data, x=x, y=y, hue=conditions)
        
        case PlotType.LINE_MEDIAN:
            if x == "Day":
                x = "Day Number"
            sns.lineplot(data=data, x=x, y=y, hue=conditions, estimator="median")

        case PlotType.SWARMPLOT:
            if x == "Day":
                x = "Day Number"
            sns.swarmplot(data=data, x=x, y=y, hue=conditions)

        case PlotType.BARPLOT:
            if x == "Day":
                x = "Day Number"
            sns.barplot(data=data, x=x, y=y, hue=conditions, estimator="mean", errorbar="sd")
            add_group_counts(ax, data=data, x=x, hue=conditions)

        case PlotType.BARPLOT_STACK:
            if x == "Day":
                x = "Day Number"
            

            order = list(data[conditions].unique())
            pivot = data.pivot(index=x, columns=[conditions, z], values=y)
            print(pivot)

            # Create a discrete colormap
            palette = custom_palette()
            cmap = mpl.colors.ListedColormap(palette)
            bounds = np.arange(0.5, len(palette) + 0.5, 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # Add colorbar
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            pivot = pivot[order]
            pivot.plot(ax=ax, kind='bar', stacked=True, color=palette, legend=False, )
            cbar = plt.colorbar(sm, ax=ax, ticks=range(len(palette)), pad=0.01)
            cbar.ax.set_title(conditions, fontsize=15)
            cbar.ax.set_yticklabels(order, fontsize=15)

            add_group_counts(ax, data=data, x=x, hue=None)

        # #Stats
        # stat_decision_tree(data, y, [x, z, *conditions])
    
    fig.tight_layout()

    #Save figure if requested
    if savefig is not None:
        fig.savefig(os.path.join(savefig, ttl+".svg"))

    plt.show()

def set_axis_label(column_name):
    """Return the label with withespace in bewteen each word and the unit."""

    name = column_name
    unit = ""

    match name:
        case "equivalent_diameter_area" | "equivalent_diameter_perimeter":
            unit = " (µm)"
        case "area_filled":
            unit = " (µm²)"
        case "centroidX", "centroidY" | "centroid_localX" | "centroid_localY":
            unit = " (pxl)"
        case "intensity_mean", "intensity_std" | "intensity_median" | "intensity_mad":
            unit = " (A.U.)"
        case "Day":
            unit = "Culture time (day)"

    name = name.replace("_", " ")
    name = capwords(name)
    name = name + unit

    return name

def check_header(variable: str, headers: List[str]):
    """Verify the `variable` string exists in `headers` list of string and raise error if not."""

    if variable != None:
        if variable not in headers:     
            raise ValueError(f"Variable {variable} do not exist.")

def add_group_counts(ax: plt.axes, data: pd.DataFrame, x: str, hue=None, y_offset=0.02, fontsize=15):
    """Add sample size (n=...) labels on top of each group in a Seaborn categorical plot.

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object returned by seaborn (e.g., from sns.violinplot, sns.boxplot, etc.)
    data : pandas DataFrame
        The data used to create the plot.
    x : str
        The column name used for the x-axis grouping.
    hue : str, optional
        The column name used for the hue grouping.
    y_offset : float, optional
        Fraction of the y-range to shift the label upward (default 0.02).
    fontsize : int, optional
        Font size of the count labels.
    """
    ylim = ax.get_ylim()
    max_y = ylim[1]
    y_shift = (ylim[1] - ylim[0]) * y_offset

    if x is None and hue is not None:
        tmp_x = hue
        hue = x
        x = tmp_x
        del tmp_x

    if hue:
        counts = data.groupby([x, hue]).size().reset_index(name='count')
        x_levels = data[x].unique()
        hue_levels = data[hue].unique()
        n_hue = len(hue_levels)

        # offsets to separate hue groups horizontally
        total_width = 0.8
        step = total_width / n_hue
        offsets = [(-total_width/2 + step/2 + i*step) for i in range(n_hue)]

        for i, x_val in enumerate(x_levels):
            for j, hue_val in enumerate(hue_levels):
                subset = counts[(counts[x] == x_val) & (counts[hue] == hue_val)]
                if not subset.empty:
                    count = subset['count'].values[0]
                    ax.text(
                        i + offsets[j],
                        max_y + y_shift,
                        f"{count}",
                        ha='center', va='bottom',
                        fontsize=fontsize
                    )
    else:
        counts = data[x].value_counts().sort_index()
        for i, (x_val, count) in enumerate(counts.items()):
            ax.text(i, max_y + y_shift, f"n={count}",
                    ha='center', va='bottom', fontsize=fontsize)

    ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.1)  # add top margin for text

def adjust_lightness_cbar(color, factor):
    r, g, b = mcolors.to_rgb(color)
    return (
        min(1, max(0, r * factor)),
        min(1, max(0, g * factor)),
        min(1, max(0, b * factor)),
    )

def custom_palette():
    """Return a custom palette of 20 colors easily differenciable and intuitive"""
    # 6 evenly spaced base colors (rainbow-like)
    base_colors = sns.color_palette("hls", 6)

    palette = [(0, 0, 0), (0.5, 0.5, 0.5)]  # start with black and gray

    for i, c in enumerate(base_colors):
        dark   = adjust_lightness_cbar(c, 0.6)
        medium = c
        light  = adjust_lightness_cbar(c, 1.4)

        if i % 2 == 0:
            # even index: dark → light
            palette.extend([dark, medium, light])
        else:
            # odd index: light → dark
            palette.extend([light, medium, dark])
    
    return palette

##################################################################################
# RESULT FILE RELATED
##################################################################################

def check_folder_results(user_inputs, resultfile="UpdatedResults.csv"):
    """Verify the folders and result files exists for execution of Plots_Stats.py script."""

    #Verify Outpath exists
    if os.path.isdir(user_inputs["Outpath"]) == False:
        logger.error(f"Oupput Path not found: {user_inputs["Outpath"]}. --> ABORTED\n\tCreate the path before runinng the script")
        raise ValueError
    
    #Verify Exp, Substrate, Day Pathe exist & Results file exist
    missing_path = []
    missing_substrates = []
    missing_days = []
    missing_result = []
    result_paths = []
    for exp,val in user_inputs["Experiments"].items():
        #Experiment path
        exp_path = val["Path"] + PATH_DELIMITER
        if os.path.isdir(exp_path) == False:
            missing_path.append(exp_path)
        else:

            #Substrate Path
            for i, s_name in enumerate(val["Substrate"]):
                s_path = os.path.join(exp_path, s_name)
                if os.path.isdir(s_path) == False:
                    missing_substrates.append(s_path)
                else:
                    
                    #Day Path
                    for d_name in user_inputs["Days"]:
                        d_path = os.path.join(s_path, d_name)
                        if os.path.isdir(d_path) == False:
                            missing_days.append(d_path)
                        else:

                            #Exact filename
                            if "*" not in resultfile:
                                result_path = os.path.join(d_path, resultfile)
                                if os.path.isfile(result_path) == False:
                                    missing_result.append(result_path)
                                else:
                                    result_paths.append({"Path": result_path, "Exp": exp, "Cond": val["Condition"][i]})
                            
                            #Look for pattern filename
                            else:
                                files_day = os.listdir(d_path)
                                pat = resultfile.replace("*", "")
                                print(val)
                                for f in files_day:
                                    print(f, pat, type(f), type(pat))
                                    file = re.search(pat, f)
                                    if file is not None:
                                        result_path = os.path.join(d_path, f)
                                        result_paths.append({"Path": result_path, "Exp": exp, "Cond": val["Condition"][i]})


    if len(missing_path) > 0:
        logger.error(f"Experiment Path not found: {"\n".join(missing_path)}")
        raise ValueError
    if len(missing_substrates) > 0:
        logger.error(f"Susbtrates Path not found: {"\n".join(missing_substrates)}")
        raise ValueError
    if len(missing_result) > 0:
        logger.error(f"Result File not found: {"\n".join(missing_result)}")
        raise ValueError
    if len(missing_days) > 0:
        logger.warning(f"Day Path not found:\n\t{"\n\t".join(missing_days)}\n\t--> Skipped.")
    
    return result_paths

def open_results(user_inputs, resultfile="UpdatedResults.csv", save: bool=False):
    """Return aggregated requested result files as pandas.DataFrame and save it as csv if requested."""
    
    data = pd.DataFrame() #data = pd.DataFrame(data=None, index=None, columns=heads, dtype=heads_type)
    
    #Verify result folder exist and store result files
    result_paths = check_folder_results(user_inputs, resultfile)
    if len(result_paths) == 0:
        logger.error("No result file found: {resultfile}. --> ABORTED")
        raise ValueError

    #Open and concatenate result files
    for f in result_paths:

        #Store result file in DataFrame
        f_data = pd.read_csv(f["Path"])

        #Add missing columns
        if "Condition" not in f_data.columns:
            f_data["Condition"] = f["Cond"]
        if "Exeriment" not in f_data.columns:
            f_data["Experiment"] = f["Exp"]
        if "Hive number" in f_data.columns:
            f_data["Hive_number"] = f_data["Hive number"]
        if "Day" in f_data.columns:
            tmp = f_data["Day"].replace("D", "", regex=True)
            f_data["Day Number"] = tmp
        if "Manually Discarded" not in f_data.columns:
            f_data["Manually Discarded"] = False

        #Select only the organoids correctly segemented
        f_data = f_data[f_data["Manually Discarded"] == False]

        #Concatenate with previous file
        if data is None:
            data = f_data
        else:
            data = pd.concat([data, f_data], ignore_index=True)

    #Verify data contains value    
    if data is None or len(data) == 0:
        logger.error(f"Data loading Failed. --> ABORTED")
        raise RuntimeError
    else:
        logger.info(f"Data loading Succes.")
    logger.debug(f"Data headers: {data.columns}")

    #Save data if requested
    if save == True:
        name = os.path.join(user_inputs["Outpath"], user_inputs["Outname"] + ".csv")
        data.to_csv(name)
        logger.info(f"Data saved: {name}.")
        
    return data

def read_graph_inputs(input_file):
    """Return, as disctionnary, user inputs used to execute Plots_Stats.py script."""

    #Verify file exists
    if os.path.isfile(input_file) == False:
        logger.error(f"User input json file do not exist: {input_file} --> ABORTED")
        raise ValueError

    #Verify it is a JSON extension
    if ".json" not in input_file:
        logger.error(f"User input file shoul be JSON file: {input_file} --> ABORTED")
        raise ValueError
    
    #Open inputs
    with open(input_file) as f:
        text = f.read()
    if "\\" in text:
        text = text.replace("\\", "\\\\")
    user_inputs = json.loads(text)

    #Verify it contains appropriate keys
    expected_keys = ["Outpath", "Outname", "Experiments", "Days"]
    missing_keys = []
    [missing_keys.append(elt) for elt in list(user_inputs.keys()) if elt not in expected_keys]
    if len(missing_keys) > 0:
        logger.error(f"Expected key(s) from json file not found: {missing_keys}. --> ABORTED.\n \
                     Expected keys are: {expected_keys}")
        raise ValueError

    #Verify it contains appropriate experiment keys
    expected_exp_keys = ["Path", "Substrate", "Condition"]
    missing_exp_keys = {elt: [] for elt in list(user_inputs["Experiments"].keys())}
    [missing_exp_keys[exp].append(i) for exp,val in user_inputs["Experiments"].items() for i in val.keys() if i not in expected_exp_keys]
    exp_rm = []
    [exp_rm.append(exp) for exp,val in missing_exp_keys.items() if len(val) == 0]
    if len(exp_rm) == 0:
        [missing_exp_keys.__delitem__(exp) for exp in exp_rm]
        logger.error(f"Expected Experiments key(s) from json file not found: {missing_exp_keys}. --> ABORTED.\n \
                     Expected Exeriment keys are: {expected_exp_keys}")
        raise ValueError
    
    logger.debug(f"User Inputs: {user_inputs}")

    return user_inputs

##################################################################################
# STATS RELATED
##################################################################################

def pretty_print(data: dict, save=False, path=None, ttl=None):
    """Nice print for statistical report and save report is `save` is True and `path` is set."""

    if save == True:
        if path is not None and ttl is not None and os.path.isdir(path):
            f = open(os.path.join(path, ttl), "w")
            print(f"============================================================\nSTATISTICAL REPORT\n{ttl}\n" \
            "============================================================\n", file=f)
        else:
            f = None
            logger.warning(f"Couldn't save Statistical report. Verify path and ttl parameters.")
    else: f is None

    for key, value in data.items():
        print(f"\n=== {key} ===")
        if f: print(f"\n=== {key} ===", file=f)
        
        if key == "stats":
            df = pd.DataFrame(value)
            print(df.to_string(index=False))
            if f: print(df.to_string(index=False), file=f)
        elif key in ["Dependant", "Independant"]:
            for group, group_value in value.items():
                print(f"\n---Test on groups: {group}---")
                if f: print(f"\n---Test on groups: {group}---", file=f)
                for sub_key, sub_value in group_value.items():
                    print(f"\t--{sub_key}--")
                    if f: print(f"\t--{sub_key}--", file=f)
                    if sub_value is None:
                        pprint(None)
                        if f: pprint(None, stream=f)
                    elif isinstance(sub_value, list):
                        df = pd.DataFrame(sub_value)
                        print(df.to_string(index=False))
                        if f: print(df.to_string(index=False), file=f)
                    else:
                        pprint(sub_value)
                        if f: pprint(sub_value, stream=f)
        else:
            pprint(value)
            if f: pprint(value, stream=f)
        print("============================================================")
        if f: 
            print("============================================================", file=f)
    
    f.close()

def compute_stats_group(data_grouped, variable, group_columns):
    
    stat_group = data_grouped.describe()
    stat_group["IQR"] = stat_group["75%"] - stat_group["25%"] #adding IQR to stats
    stat_group["skew"] = data_grouped.skew()[variable] #adding skewness to stats
    stat_group["pct outlier"] = 0
    stat_group["Any(|Zscore|>3)"] = False
    stat_group["Any(value>3*std)"] = False
    i=0
    for name, group in data_grouped:
        below_5pct = (group < (stat_group["25%"][i] - 1.5*stat_group["IQR"][i])).value_counts().get(True)
        if below_5pct is None:
            below_5pct = 0
        above_95pct = (group > (stat_group["75%"][i] + 1.5*stat_group["IQR"][i])).value_counts().get(True)
        if above_95pct is None:
            above_95pct = 0
        stat_group.loc[i, "pct outlier"] = 100*(below_5pct + above_95pct) / stat_group["count"][i]
        stat_group.loc[i, "Any(|Zscore|>3)"] = any(abs(z) > 3 for z in zscore(group))
        stat_group.loc[i, "Any(value>3*std)"] = any((group > 3*stat_group.loc[i, "std"]))
        i+=1
    stat_group["group"] = stat_group[group_columns].apply(tuple, axis=1)
    logger.debug(f"Statistical table: \n\t: {stat_group}")

    return stat_group

def setting_name_pair_unpair_column(group_columns, day_dependance):

    paired_column = []
    if "Day" in group_columns:
        if day_dependance is None:
            logger.error("If 'Day' is used to group variable, 'day_dependance' should be set: either 'match' or 'anonymous'")
            raise KeyError
        elif day_dependance == 'match':
            paired_column.append("Day")
            paired_column_index = group_columns.index("Day")
            if paired_column_index != 0:
                logger.error("If 'Day' is used to group variable, 'Day' should be the first element of group_columns")
                raise KeyError
            unpaired_column = group_columns.copy()
            unpaired_column.remove("Day")
        elif day_dependance == 'anonymous':
            unpaired_column = group_columns.copy()
        else:
            logger.error(f"{day_dependance} key unkown for 'day_dependance'. Available keys are: 'match' and 'anonymous'.")
            raise KeyError
        
        return paired_column, unpaired_column

def structure_pair_unpair_group_name(data, group_columns, paired_groups, unpaired_groups):

    real_groups = list(data.groupby(by=group_columns).groups.keys())
    tmp_unpaired = [list(itertools.product([pairs], unpaired_groups)) for pairs in paired_groups]

    unpaired_groups_all = []
    for group_name in tmp_unpaired:
        unpaired_groups_all.append([sum(elt, ()) for elt in group_name if sum(elt, ()) in real_groups])
    tmp_paired = [list(itertools.product(paired_groups,[pairs])) for pairs in unpaired_groups]

    paired_groups_all = []
    for group_name in tmp_paired:
        paired_groups_all.append([sum(elt, ()) for elt in group_name if sum(elt, ()) in real_groups])
    
    logger.debug(f"Unpaired groups: {unpaired_groups_all}")
    logger.debug(f"Paired groups: {paired_groups_all}")

    return paired_groups_all, unpaired_groups_all

def conservative_stat_decision_tree(data: pd.DataFrame, variable: str, group_columns: List[str], variable_type: str, what: str, day_dependance: str=None, force_parametric=False):
    """Decide what test to perform, generate report and returns associated p-values. All groups are considered as independant expect for `Day` if `day_dependance` is set to `match`.
    
    Parameter:
    -----------
    data : pandas DataFrame.
        Dataframe containing as header at least the variable to look at and the group_columns used to group the variable.    
    variable : string.
        Name of the variable to look at. The exact same string should be a header from data.
    group_column : List of string.
        On what you want to group the variable. Each element of group_column should be a header from data.
    variable_type : str {'continuous', 'ordinal', 'nominal'}.
        The type of data you want to look at:
        - 'continuous': variable that can have whatever value. It is usually the case for all data measured using an instrument.
        - 'ordinal': categorical values that can be sorted (ex: )
        - 'nominal': categorical values that cannot be sorted (ex: colors)
    what : str {'difference', 'relationship'}
        What you want to look at on your data:
        - 'difference': want to determine if there is a significant difference in one variable of your samples
        - 'relationship': want to determine if there is a relationship between 2 variables of your samples
    day_dependance : str {'match', 'anonymous'}
        If 'Day' is one of your grouping variable (ie, one element of group_column list), defines if 'Day' should be considered as dependant or independant variable.
        - 'match': all of your samples have match paires --> 'Day' is considered as a dependant variable
        - 'anonymous': your samples is not matching paires --> 'Day' is considered as an independant variable
    force_parametric : bool. False by default
        Set to True if you want to perform a parametric test eventhough the conditions to satisfy parametric test are not reached.

    Return:
    --------
    results : dict
        Result of the statistical tests for each 2group combination.
        ```
        results = {
            "what":
            "variable":
            "conditions":
            "variable_type":
            "day_dependance":
            "force_parametric":
            "stats":
            "Parametric":
            "Dependant":
            {
                "Test on groups: XX":
                {
                    "Global": if more than 2 groups
                    [{
                        "Ngroup":
                        "Global Test":
                        "P-value":
                    }]
                    "Comparisons":
                    [
                        {
                            "Test":
                            "Group":
                            "P-value":
                        },

                    ]
                }
            }
            }
        }
        ```

    Representation of decisison tree:
    ---------------------------------
    ```
    START
    │
    ├─ Objective
    │   ├─ Difference → continue
    │   └─ Relationship → Pearson if param-eligible, else Spearman
    │
    ├─ Data type
    │   ├─ Nominal → Chi² / Fisher / McNemar / Cochran Q → END
    │   ├─ Ordinal → Wilcoxon / Mann–Whitney / Friedman / Kruskal → END
    │   └─ Continuous → continue
    │
    ├─ Determine:
    │   ├─ Paired vs Independent
    │   ├─ 2 groups vs >2 groups
    │   ├─ n per group
    │   ├─ Skewness per group
    │   ├─ Outliers (1.5*IQR)
    │   └─ Extreme z > |3.5|
    │
    ├─ Parametric eligibility:
    │   IF (ALL groups satisfy):
    │       |skew| ≤ 2
    │       AND <5% outliers
    │       AND |Zscore|>3
    │   → PARAMETRIC
    │      ├─ 2 groups:
    │      │     Paired → Paired t-test
    │      │     Indep → t-test (Welch if unequal variance)
    │      │
    │      └─ >2 groups:
    │            Paired → RM-ANOVA → Paired t-test for each combination with Holm p-values adjustment
    │            Indep → ANOVA (Welch if unequal variance) → Tuckey for each combination (or Games-Howek if unequal variance)
    │
    └─ ELSE → NONPARAMETRIC
            ├─ 2 groups:
            │     Paired → Wilcoxon (Sign if very small n)
            │     Indep → Mann-Whitney
            └─ >2 groups:
                Paired → Friedman → Dunn Test with Holm Corrections
                Indep → Kruskal-Wallis → 
    ```
    """

    # What effect size to do and when:
    # 2 groups parametric → Hedges’ g
    # more than 2 groups → Omega squared (ω²)
    # Nonparametric → Cliff’s Delta
    # Correlation → report r or rho
    
    #Verify required headers are present
    for elt in (*group_columns, variable):
        check_header(elt, data.columns.values.tolist())
    if None in group_columns:
        group_columns.remove(None)

    #Initiate output
    stats = {"what": what, "variable": variable, "conditions": group_columns, 
             "variable_type": variable_type, "day_dependance": day_dependance, 
             "force_parametric": force_parametric}

    #Setting names of paired & unpaired columns from data
    paired_column, unpaired_column = setting_name_pair_unpair_column(group_columns, day_dependance)

    #Setting the group names of paired & unpaired only
    unpaired_names = [list(set(data[elt])) for elt in unpaired_column]
    paired_names = [list(set(data[elt])) for elt in paired_column]
    unpaired_groups = list(itertools.product(*unpaired_names))
    paired_groups = list(itertools.product(*paired_names))

    #Setting and structuring the group names of paired and unpaired groups
    paired_groups_all, unpaired_groups_all = structure_pair_unpair_group_name(data, group_columns, paired_groups, unpaired_groups)

    if what == 'difference':

        if variable_type == 'ordinal':
            logger.error(f"Not implemented yet for variable_typ={variable_type}")
            raise KeyError

        elif variable_type == "nominal":
            logger.error(f"Not implemented yet for variable_typ={variable_type}")
            raise KeyError

        elif variable_type == 'continuous':
            logger.debug(f"variable_type={variable_type}")
            
            #Group data
            data_grouped = data.groupby(by=group_columns, as_index=False)[variable]
            data_grouped_as_index = data.groupby(by=group_columns, as_index=True)

            #Compute basic stat on groups (mean, median, ...)
            stat_group = compute_stats_group(data_grouped, variable, group_columns)
            #Store it into output
            stats["stats"] = stat_group.to_dict() #store statistical info
                
            #Paired Test
            if len(paired_column) > 0:

                stats["Dependant"] = {}
                
                for group in paired_groups_all:

                    #Go to next iteration is one group do not exist
                    if any(elt not in data_grouped_as_index.groups.keys() for elt in group) or len(group) == 0:
                        continue

                    #Determine if the groups statifies conditions for parametric tests
                    sub_stat_group = stat_group.loc[stat_group["group"].isin(group)]
                    if all(abs(sub_stat_group["skew"]) < 2) and all(sub_stat_group["Any(|Zscore|>3)"] < 3) and all(sub_stat_group["pct outlier"] < 5):# and all(sub_stat_group["count"] >= 25):
                        parametric = True
                    else:
                        parametric = False
                    logger.debug(f"Paired Test\n\tGroup={group}\n\tParametric={parametric}")

                    #Parametric Test
                    if  parametric == True or force_parametric == True: #add the condition on outliers and extrema

                        stats["Dependant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}                       
                        
                        #Only two groups ===> PAIRED T-TEST
                        if len(group) <= 2:    
                            data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')
                            _, pval = ttest_rel(*data_group)    
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 
                            stats["Dependant"][str(group)]["Comparisons"] = [{"Test": "Paired T-test", "Comparison": group, "P-value": pval,
                                                                              "N per group": n, "Warning": warn_mes}]     

                        #More than two groups ===> REPEATED MEASURES ANOVA
                        else:
                            data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'dataframe')
                            aov = AnovaRM(data_group, variable, "Hive_number", within=["Group_Name"]).fit()
                            pval_global = aov.anova_table['Pr > F']["Group_Name"]
                            stats["Dependant"][str(group)]["Global"] = [{"Ngroup": len(group), "Global Test": "Repeated Measures ANOVA", "P-value": pval_global}]
                            if pval_global < 0.05:
                                stats["Dependant"][str(group)]["Comparisons"] = []
                                group_combination = list(itertools.combinations(group, 2))
                                grouped = data_group.groupby(by="Group")
                                pval_no_adjust = []
                                for g in group_combination:
                                    gg = structure_data(grouped, g, variable, 'array_like')
                                    _, p = ttest_rel(*gg)
                                    pval_no_adjust.append(p)
                                pval = multipletests(pval_no_adjust, method='holm')
                                for i, g in enumerate(group_combination):
                                    n = (sub_stat_group.loc[sub_stat_group["group"] == g[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == g[1]]["count"].values[0])
                                    if any(elt < 15 for elt in n):
                                        warn_mes = "Less than 15 elements"
                                    else:
                                        warn_mes = None 
                                    stats["Dependant"][str(group)]["Comparisons"].append({
                                        "Test": "Paired T-test with Holm correction",
                                        "Group": g,
                                        "P-value": pval_no_adjust[i],
                                        "Adjusted P-values": pval[1][i],
                                        "N per group": n,
                                        "Warning": warn_mes
                                    })
                    
                    #Non parametric Test
                    else:

                        stats["Dependant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}
                        data_group = structure_data(data_grouped_as_index, group, variable, 'array_like')
                        
                        #Only two groups
                        if len(group) <= 2:    
                            
                            #The two groups have at least 10 elements each ===> WICOXON SIGNED RANK
                            if all(stat_group["count"] >= 10):
                                _, pval, _ = wilcoxon(*data_group)
                                test = "Wilconxon Signed Rank Test"
                        
                            #At least one group has less than 10 elements ===> SIGN TEST
                            else:
                                pval=None; test="Not implemented"

                            #Verify number of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 

                            #Update results
                            stats["Dependant"][str(group)]["Comparisons"] = [{"Test": test, "Group": group, "P-value": pval,
                            "N per group": n, "Warning": warn_mes}]

                        #More than two groups ===> FRIEDMAN for global, then DUNN TEST and HOLM CORRECTIONS
                        else:
                            #Initiaing pair-wise outputs
                            gg = None
                            pval = None
                            #Perform Global Test: Friedman
                            _, pval_global = friedmanchisquare(*data_group)
                            stats["Dependant"][str(group)]["Global"] = [{"Ngroup": len(group), "Global Test": "Friedman", "P-value": pval_global}]
                            #Perform pair-wise tests: Dunn with Holm correcttion
                            if pval_global < 0.05:
                                dunn = posthoc_dunn(data_group, p_adjust='holm')
                                gg = []; pval=[]
                                stats["Dependant"][str(group)]["Comparisons"] = []
                                for i in dunn:
                                    for j in dunn.index:
                                        if j > i:
                                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[i-1]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j-1]]["count"].values[0])
                                            if any(elt < 15 for elt in n):
                                                warn_mes = "Less than 15 elements"
                                            else:
                                                warn_mes = None 
                                            stats["Dependant"][str(group)]["Comparisons"].append({"Test": "Dunn Test with holm corrections", 
                                                                                                  "Group":(group[i-1], group[j-1]), "P-value": dunn[i][j],
                                                                                                  "N per group": n, "Warning": warn_mes})

            #Unpaired Test
            if len(unpaired_column) > 0 and len(unpaired_groups) > 1:

                stats["Independant"] = {}

                for group in unpaired_groups_all:

                    if any(elt not in data_grouped_as_index.groups.keys() for elt in group) or len(group) == 0:
                        continue

                    print("Independant", group)

                    #Determine if the groups statifies conditions for parametric tests
                    sub_stat_group = stat_group.loc[stat_group["group"].isin(group)]
                    if all(abs(sub_stat_group["skew"]) < 2) and all(sub_stat_group["Any(|Zscore|>3)"]) < 3 and all(sub_stat_group["pct outlier"] < 5): #and all(sub_stat_group["count"] >= 25)
                        parametric = True
                    else:
                        parametric = False
                    logger.debug(f"Unpaired Test\n\tGroup={group}\n\tParametric={parametric}")

                    stats["Independant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}

                    #Parametric Test
                    if parametric == True or force_parametric == True: 

                        data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')
                        _, variance = bartlett(*data_group)

                        #Only two groups
                        if len(group) <= 2:

                            #Equal variance ===> UNPAIRED T-TEST
                            if variance >= 0.05:
                                _, pval = ttest_ind(*data_group, equal_var=True)  
                                test = "Unpaired T-Test"    

                            #Unequal variance ===> WELCH T-TEST
                            else:
                                _, pval = ttest_ind(*data_group, equal_var=False)  
                                test = "Welch T-Test"  

                            #Determine number of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 

                            #Update result
                            stats["Independant"][str(group)]["Comparisons"] = [{
                                "Variance Test": "Barlett", 
                                "Variance P-Value": variance, 
                                "Group": group, "P-value": pval, "Test": test,
                                "N per group": n, "Warning": warn_mes}]   

                        #More than two groups: test variance
                        else:

                            #Equal variance ===> ONE-WAY ANOVA
                            if variance >= 0.05:
                                glob= "One-way ANOVA"
                                _, pval_global = f_oneway(*data_group, equal_var=True)
                                if pval_global < 0.05:
                                    tuk = tukey_hsd(*data_group, equal_var=True)
                                    test = "Pair-wise Tukey"
                                    stats["Independant"][str(group)]["Comparisons"] = []
                                    for i, elt in enumerate(tuk.pvalue):
                                        for j, p in enumerate(elt):
                                            if j > i:
                                                #Determine numebr of elements per group
                                                n = (sub_stat_group.loc[sub_stat_group["group"] == group[i]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j]]["count"].values[0])
                                                if any(elt < 15 for elt in n):
                                                    warn_mes = "Less than 15 elements"
                                                else:
                                                    warn_mes = None 
                                                #Update results
                                                stats["Independant"][str(group)]["Comparisons"].append({"Test": test, "Group": (group[i], group[j]), "P-value": p,
                                                                                                        "N per group": n, "Warning": warn_mes})    

                            #Unequal variance ===> WELCH ANOVA
                            else:
                                glob= "Welch ANOVA"
                                _, pval_global = f_oneway(*data_group, equal_var=False)
                                if pval_global < 0.05:
                                    tuk = tukey_hsd(*data_group, equal_var=False) 
                                    test = "Games-Howell"
                                    stats["Independant"][str(group)]["Comparisons"] = []
                                    for i, elt in enumerate(tuk.pvalue):
                                        for j, p in enumerate(elt):
                                            if j > i:
                                                #Determine numebr of elements per group
                                                n = (sub_stat_group.loc[sub_stat_group["group"] == group[i]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j]]["count"].values[0])
                                                if any(elt < 15 for elt in n):
                                                    warn_mes = "Less than 15 elements"
                                                else:
                                                    warn_mes = None 
                                                #Update results
                                                stats["Independant"][str(group)]["Comparisons"].append({"Test": test, "Group": (group[i], group[j]), "P-value": p,
                                                                                                        "N per group": n, "Warning": warn_mes})                            

                            stats["Independant"][str(group)]["Global"] = [{"Ngroup": len(group), "Variance Test": "Bartlett", "Variance P-value": variance,
                                                                "Global Test": glob, "P-value": pval_global}]

                    #Non-parametric Test
                    else:

                        data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')

                        #Only two groups ===> MANN-WHITNEY
                        if len(group) <= 2:
                            #Statistic Test  
                            _, pval = mannwhitneyu(*data_group)
                            #Determine numebr of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 
                            #Update results
                            stats["Independant"][str(group)]["Comparisons"] = [{
                                "Test": "Mann-Whitney U", "Group": group, "P-value": pval,
                                "N per group": n, "Warning": warn_mes}]   

                        #More than two groups ===> KRUSKAL-WALLIS for global, than DUNN TEST and HOLM CORRECTIONS
                        else:
                            _, pval_global = kruskal(*data_group)
                            stats["Independant"][str(group)]["Global"] = [{
                                "Ngroup": len(group), "Test": "Kruskal-Wallis", "P-value": pval_global}]
                            if pval_global <0.05:
                                data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'dataframe')
                                dunn = posthoc_dunn(data_group, val_col=variable, group_col="Group_Name", p_adjust='holm')
                                stats["Independant"][str(group)]["Comparisons"] = []
                                for ii, i in enumerate(dunn):
                                    for jj, j in enumerate(dunn.index):
                                        if j > i:
                                            #Determine numebr of elements per group
                                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[ii-1]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[jj-1]]["count"].values[0])
                                            if any(elt < 15 for elt in n):
                                                warn_mes = "Less than 15 elements"
                                            else:
                                                warn_mes = None 
                                            #Update results
                                            stats["Independant"][str(group)]["Comparisons"].append({"Test": "Dunn Test with Holm corrections", 
                                                                                                  "Group":(group[ii-1], group[jj-1]), "P-value": dunn[i][j],
                                                                                                  "N per group": n, "Warning": warn_mes})   
    
        else:
            logger.error(f"Key do not exist for variable_type={variable_type}")
            raise KeyError

    elif what == 'relationship':
        logger.error(f"Not implemented yet for what={what}")
        raise KeyError
    
    else:
        logger.error(f"Key do not exist for what={what}")
        raise KeyError
    
    ttl = f"StatisticalReport__var{variable}__group{"-".join(group_columns)}.txt"

    return stats, ttl

def structure_data(data_grouped_as_index, group, variable, request):

    #print(data_grouped_as_index, group, variable, request)

    if request == 'array_like':
        data_group = []
        for g in group:
            if g in data_grouped_as_index.groups.keys():
                values = data_grouped_as_index.get_group(g)[variable]
                order = data_grouped_as_index.get_group(g)["Hive_number"]
                sorted_values = [val for _, val in sorted(zip(order, values))]
                data_group.append(sorted_values)

    elif request == 'dataframe':
        data_group = None
        for g in group:
            if g in data_grouped_as_index.groups.keys():
                tmp = pd.DataFrame({
                    "Group": [g for _ in range(len(data_grouped_as_index.get_group(g)))],
                    "Group_Name": ["_".join(g) for _ in range(len(data_grouped_as_index.get_group(g)))],
                    "Hive_number":  data_grouped_as_index.get_group(g)["Hive_number"],
                    variable: data_grouped_as_index.get_group(g)[variable]})
                if data_group is None:
                    data_group = tmp
                else:
                    data_group = pd.concat([data_group, tmp])     

    return data_group

##################################################################################
# FITS RELATED
##################################################################################

def gauss_fit(x, *params):
    """Apply gaussian function on x values using the params as parameters"""

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def intensity_mad(regionmask, intensity_image):
    return median_abs_deviation(intensity_image[regionmask])

def intensity_median(regionmask, intensity_image):
    return np.median(intensity_image[regionmask])
