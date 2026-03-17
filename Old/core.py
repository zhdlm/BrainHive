from __future__ import annotations  # recommended by chatgpt
import os
from datetime import datetime
import warnings
import itertools
import csv
import numpy as np
from typing import List
from enum import Enum
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rotate, resize
from skimage.draw import line, disk
from skimage.morphology import binary_erosion, binary_dilation, ellipse, area_closing
from skimage.filters import threshold_otsu
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops
from scipy.signal import fftconvolve, find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
from math import floor, sqrt, gcd, ceil

#####
# TO DO:
#    -Crop methods for levels:
#        -Substrate
#        -Exp
#    -Optimized crop methods for levels:
#        -Substrate
#        -Exp
#    -Threshold methods for levels:
#        -Substrate
#        -Exp
#    -Organoid mask methods
#        -Substrate
#        -Exp
#    -Save organoid results method
#        -Substrate
#        -Exp
# 
#####

VERSION = "v0.1"

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

scale_ratio=10

def simple_warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"
warnings.formatwarning = simple_warning_format

def get_all_attribute_names(obj):
    names = set(vars(obj).keys())  # instance attributes

    # Add property names
    for name, value in obj.__class__.__dict__.items():
        if isinstance(value, property):
            names.add(name)

    return sorted(names)

def image_number_from_filename(filename, format='int'):
    """Returns the number written in the image as int by default or in string if format=='str'."""
    pattern = "00[0-1][0-9]"
    number = re.search(pattern, filename).group()
    if format == 'int':
        return int(number)
    elif format == 'str':
        return str(number)
    else:
        raise ValueError(f"format unexcpected: either 'int' or 'str'.")
    
def idx_acquisition_order(acquisition_order):
    """"""

    # Substrate is definied as follow:
    # 1st column: 3 hives
    # 2nd column: 4 hives
    # 3rd column: 5 hives
    # 4th column: 3 hives
    # 5th column: 4 hives
    
    idx = None

    match acquisition_order:
        case 'bottom':
            idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]] #old version
        case 'bottom-flip':
            idx = [[16, 15, 7], [17, 14, 8, 6], [18, 13, 9, 5, 0], [12, 10, 4, 1], [11, 3, 2]] #old version
        case 'bottom right':
            idx = [[2,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]] #old version
        case 'exp001':
            idx = [[7,6,5], [15,8,4,0], [16,14,9,3,1], [17,13,10,2], [18,12,11]] #old version
        # case 'bottom center':
        #     pass
        #     idx = [[,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]] #old version
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

def hexagon_outline_ndarray(image_shape, center, side, thickness=1, return_type='hexagon'):
    """
    Draws a hexagon outline (not filled) into a 2D ndarray.
    
    Parameters:
        image_shape (list): [height, width] of the output ndarray.
        center (list): [y, x] center of the hexagon.
        side (float): side length of the hexagon.

    Returns:
        np.ndarray: ndarray with hexagon outline written as 1s.
    """
    # print('Drawing hexagon outlines: image size=', image_shape, 'center=', 
    #       center, 'side=', side, 'thickness=', thickness)

    height = image_shape[0]
    width = image_shape[1]
    if return_type == 'hexagon':
        img = np.zeros((height, width), dtype=np.int8)
    elif return_type == 'side':
        img = [np.zeros((height, width), dtype=np.int8) for _ in range(6)]
    elif return_type == 'coord-side':
        img = []

    cx = center[1]
    cy = center[0]
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    x_vertices = cx + side * np.cos(angles)
    y_vertices = cy + side * np.sin(angles)

    # Draw lines between consecutive vertices
    for i in range(6):
        x0, y0 = int(round(x_vertices[i])), int(round(y_vertices[i]))
        x1, y1 = int(round(x_vertices[(i + 1) % 6])), int(round(y_vertices[(i + 1) % 6]))
        rr, cc = line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
        # For each pixel in the line, draw a small disk of radius thickness//2
        if return_type == 'coord-side':
            img.append([x0,y0])
            img.append([x1, y1])
        else:
            for r, c in zip(rr, cc):
                if return_type == 'hexagon':
                    dr, dc = disk((r, c), radius=thickness // 2, shape=img.shape)
                    img[dr, dc] = 1
                elif return_type == 'side':
                    dr, dc = disk((r, c), radius=thickness // 2, shape=img[0].shape)
                    img[i][dr, dc] = 1

    del(image_shape, center, side)
    return img

def assign_dim(design, microscope, magnification, dimension):
    if design in ["H2M100", "H2M150", "H2M200"]:
        height = 2000
        n_expect = 19
    elif design == 1:
        height = 1000
        n_expect = 19
    elif design == 0: #for test
        height = 2000
        n_expect = 1000

    match microscope:

        case "Leica":
            match magnification:
                case "x5":
                    if dimension == "1944x2592":
                        um2pixel = 0.9 # 1 µm corresponds to 0.5645 pxl
                        pxl2um = 1.1 # 1 pxl corresponds to 1.7714 µm
                    elif dimension == "1200x1600":
                        um2pixel = 0.54 # 1 µm corresponds to 0.5645 pxl
                        pxl2um = 1.85 # 1 pxl corresponds to 1.7714 µm
    
    return height, n_expect, um2pixel

def adjust_dimension_after_rotation(image, hive_dimension, image_type):
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
            tmp = np.empty((hive_dimension[i], image.shape[i], 3), dtype=image_type)
            tmp[round(w/2):round(image.shape[i]+w/2), :, :] = image
            image_out = tmp
    
    del(image, hive_dimension, image_type)
    return image_out

def add_empty_img(images, file_name, filename_pattern, n_expect):
    
    out_images = images.copy()
    out_file_name = file_name.copy()
    
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

def gauss_fit(x, *params):
    """Apply gaussian function on x values using the params as parameters"""

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

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

def get_img_center(img_dim, axes=1):
    """Return the (y,x) coordinates of image center."""
    if axes == 1:
        return (round(img_dim[0]/2), round(img_dim[1]/2))
    elif axes == 0:
        return (round(img_dim[1]/2), round(img_dim[0]/2))
    
def display_montage_organoids_dynamic(hierarchy: dict, obj: List[Exp], days: List[str], image_type: ImageType, ext=None):
    """
        hierachy = {
            expI: {
                substrateI: ["hive_number_I", "hive_number_I", "hive_number_3"],
            }
        }
    """
    #Verify the experiment names are consistent and exist
    obj_names = [elt.name for elt in obj]
    hierarchy_lvl_1 = list(hierarchy.keys())
    if bool(set(hierarchy_lvl_1) & set(obj_names)) == False:
        raise ValueError(f"Experiment named {"-".join(hierarchy_lvl_1)} in hierarchy do not exist for experiment {obj_names}")
    
    #Create hierarchy from obj
    hierarchy_obj = {}
    for exp in obj:
        hierarchy_obj[exp.name] = {}
        for idx_s_name, s_name in enumerate(exp.substrates_names):
            hierarchy_obj[exp.name][s_name] = {}
            for idx_d_name, d_name in enumerate(exp.substrates_refs[idx_s_name].days_names):
                hierarchy_obj[exp.name][s_name][d_name] = exp.substrates_refs[idx_s_name].days_refs[idx_d_name].hive_number
    
    #Create real hierarchy and determine dimension
    hierarchy_real = hierarchy.copy()
    exp_names_real = []
    substrate_names_real = []
    day_names_real = []
    hives_number_real = []
    cond = []
    for exp in hierarchy_lvl_1:
        if exp not in list(hierarchy_obj.keys()):
            del hierarchy_real[exp] #Remove experiment dictionnary in hieracy if do not exist in Exp instances
        else:
            exp_names_real.append(exp)
            for s in list(hierarchy[exp].keys()):
                if s not in list(hierarchy_obj[exp].keys()):
                    del hierarchy_real[exp][s] #Remove susbtrate dictionnary in hieracy if do not exist in Exp instances
                else:
                    substrate_names_real.append(s) #Update the real values for substrate names (ie intersection of hierarchy and Exp instances)
                    missing_hives = [] 
                    day_count = 0
                    for d in days:
                        if d in list(hierarchy_obj[exp][s].keys()):
                            day_names_real.append(d) #Update the real values for day names (ie intersection of hierarchy and Exp instances)
                            day_count += 1
                            #Store all hives (for a given substrate and for each day) that are not associated with picture
                            for h in hierarchy[exp][s]:
                                if h not in hierarchy_obj[exp][s][d]:
                                    missing_hives.append(h)                
                    for n in set(missing_hives):
                        if missing_hives.count(n) == day_count:
                            hierarchy_real[exp][s].remove(n) #Remove hive element in hieracy if do not exist in Exp instances for this specific substrate
                    hives_number_real += hierarchy_real[exp][s] #Update the real values for hive numbers (ie intersection of hierarchy and Exp instances)
                    #Create list of condition names that will be written on the graphic
                    prefix_cond = exp + "\n" + s + "\n"                    
                    cond += (list(map(lambda x: prefix_cond + x, map(str,hierarchy_real[exp][s]))))
    day_names_real = list(set(day_names_real))
    day_names_real = [d for d in days if d in day_names_real] #to sort the days
    n_hives = len(hives_number_real)
    n_days = len(day_names_real)
    #print(hierarchy, "\n", hierarchy_real)

    #Determine grid orientation
    if n_hives < n_days:
        nrows = n_hives
        ncols = n_days
        ttl_subplot = day_names_real
        ylabel = cond
    else:
        nrows = n_days
        ncols = n_hives
        ylabel = day_names_real
        ttl_subplot = cond
    #Create figure and axes
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    #Assign title to each subplot
    for c in range(ncols):
        ax[0, c].set_title(ttl_subplot[c])
    #Assign ylabel to each subplot
    for r in range(nrows):
        ax[r, 0].set_ylabel(ylabel[r], labelpad=100, rotation=0, va='center')
    #Remove ticks and tick labels
    for axx in ax.flat:
        axx.set_xticks([])
        axx.set_yticks([])
    #Remove space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    for axx in fig.axes:
            text = []
            for txt in axx.texts: text.append(txt)
            for item in (
                axx.title,
                axx.xaxis.label,
                axx.yaxis.label,
                *axx.get_xticklabels(),
                *axx.get_yticklabels(),
                *text
            ):
                item.set_fontsize(20)

    #Create empty image
    hive_tmp = obj[0].substrates_refs[0].days_refs[0].hives_refs[0]
    filename = image_type.value + hive_tmp.filename_raw
    if ext is not None:
        filename_ext = filename.split(".")
        if len(filename_ext) == 2 :
            filename_ext = filename_ext[1]
        else:
            raise ValueError(f"{filename} filename is not usable.")
        filename = filename.replace(filename_ext, ext)
    img_tmp = hive_tmp.open_image(rgb=True, filename=filename)
    dim = img_tmp.shape
    img_empty = np.zeros(dim)
    del hive_tmp, filename, img_tmp, dim
    
    #Open picture & put in in grid
    i = 0
    for exp in hierarchy_real:
        idx_e = obj_names.index(exp)
        for s in hierarchy_real[exp]:
            idx_s = obj[idx_e].substrates_names.index(s)
            for h in hierarchy_real[exp][s]:
                for d in day_names_real:
                    #if d in list(hierarchy_obj[exp][s].keys()):
                    if n_hives < n_days:
                        c = day_names_real.index(d)
                        r = i % nrows
                    else:
                        r = day_names_real.index(d)
                        c = i % ncols
                    if d in list(hierarchy_obj[exp][s].keys()) and h in hierarchy_obj[exp][s][d]:
                        idx_d =  obj[idx_e].substrates_refs[idx_s].days_names.index(d)
                        idx_h = obj[idx_e].substrates_refs[idx_s].days_refs[idx_d].hive_number.index(h)
                        hive = obj[idx_e].substrates_refs[idx_s].days_refs[idx_d].hives_refs[idx_h]
                        filename = image_type.value + hive.filename_raw
                        #print(exp, s, d, h, filename, r, c)
                        img = hive.open_image(rgb=True, filename=filename)
                    else:
                        img = img_empty
                    ax[r,c].imshow(img)
                i += 1                      

    plt.show()

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

        print(f'Click at location {event.xdata}, {event.ydata}, {type(event.xdata)}')

        for i,elt in enumerate(coords): #loop over the organoid list

            if elt["Hive coord"][0][0] <= event.xdata < elt["Hive coord"][0][1] and elt["Hive coord"][1][0] <= event.ydata < elt["Hive coord"][1][1]: #Check if click is in hive

                #Generate a sub_list of organoids of this specific hive
                sub_list =  []
                [sub_list.append(elt) for j in range(len(coords)) if coords[j]["Hive number"] == elt["Hive number"]]
                print(f"Organoids inside hives: {sub_list}")

                for sub_elt in sub_list:

                    #Coordinates of the organoid in the montage image
                    xmin = round(sub_elt["bbox"][1] + elt["Hive coord"][0][0])
                    xmax = round(sub_elt["bbox"][3] + elt["Hive coord"][0][0])
                    ymin = round(sub_elt["bbox"][0] + elt["Hive coord"][1][0])
                    ymax = round(sub_elt["bbox"][2] + elt["Hive coord"][1][0])

                    if  xmin <= event.xdata < xmax and ymin <= event.ydata < ymax: # Check if click is on organoid
                        print(f"Selected this organoid: {sub_elt}")

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
                        print(f"Updated list of driscarded organoids: {discarded_organoids}")
                        
                        break #to exit loop if organoid is found

        
        event.canvas.draw_idle() #Update plot

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
    OVERLAY = "Overlay_"
    MONTAGE = "Montage_"

    @classmethod
    def get_attr(cls):
        return [elt.value for elt in cls]
    
class OrganoidProperties(Enum):
    props = ['label', 'centroidX', 'centroidY', 'centroid_localX', 'centroid_localY', 'bbox_ymin', 'bbox_xmin', 'bbox_ymax', 'bbox_xmax', 
         'eccentricity', 'area_filled', 'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length',
         'intensity_mean', 'intensity_std',
         'image_filled', 'image_intensity']
    metadata = ['Index', 'Day', 'Substrate', 'Image Name', 'Hive number', 'Manually Discarded','Analysis Date', 'Analysis Version']

class HexagonMath:

    def __init__(self, height):

        self.height = height
        self.side = round(self.height/(2*np.sin(np.pi/3)))
        self.diagonal = 2*self.side

    @classmethod
    def get_side_from_height(cls, height):
        return round(height/(2*np.sin(np.pi/3)))
    
    @classmethod
    def get_diagonal_from_height(cls, height):
        return round(2*height/(2*np.sin(np.pi/3)))
    
    @classmethod
    def hexagon_outline_ndarray(image_shape, center, side, thickness=1, return_type='hexagon'):
        pass

class Exp:

    substrates_refs: list[Substrate] #enable autocompletion and will hold Substrate instances
    
    def __init__(self, name: str, path: str, days: List[str], substrates: List[str], filename_pattern: str, params: List[dict], start_date: str=None, protocol_path: str=None):
        
        self.name = name
        self.path = path
        self.filename_pattern = filename_pattern
        self.start_date = start_date
        self.protocol_path = protocol_path
        self.substrates_names = substrates
        self.days_names = days
        self.filenames = [[None for _ in range(len(days))] for _ in range(len(substrates))]
        self.params = params
        self.substrates_refs = []

        #Verify path existence
        if os.path.isdir(self.path) == False:
            raise ValueError("Experiment folder do not exist.")
        
        #Create and store daughter class
        for s in substrates:
            substrate = Substrate(s, self, days, params)
            self.substrates_refs.append(substrate)

        # #Display hierarchy
        # self.display_hierarchy()

    def __repr__(self):
        days = ",".join(self.days_names)
        substrates = ",".join(self.substrates_names)
        rep = "Exp(" + self.name + ", " + self.path + " , [" + substrates + "], [" + days + "], " + str(self.start_date) + ", " + str(self.protocol_path) +")"
        return rep
    
    def display_hierarchy(self):
        print(f"{self}:")
        for s in self.substrates_refs:
            print(f"\t{s}")
            for d in s.days_refs:
                print(f"\t\t{d}")

    def add_susbtrate(self, substrate_name):
        if substrate_name not in self.substrates_names:
            self.substrates_names.append(substrate_name) #update list of names
            substrate = Substrate(substrate_name, parent=self, days=self.days_names) #create Substrate class instance
            self.substrates_refs.append(substrate) #update ref list
    
    def rm_substrate(self, substrate_name):
        if substrate_name in self.substrates_names:
            idx = self.substrates_names.index(substrate_name) #update list of names
            del(self.substrates_names[idx], self.substrates_refs[idx]) #update list of refs

    #Here as is needs to be apply to all experiment
    def add_day(self, day_name):
        if day_name not in self.days_names:
            self.days_names.append(day_name) #update list of names
            for s in self.substrates_refs:
                day = Day(day_name, parent=s) #create Day class instance
                #s.days_names.append(day_name) #update susbtrate class
                s.days_refs.append(day)

    #Here as is needs to be apply to all experiment
    def rm_day(self, day_name):
        if day_name in self.days_names:
            idx = self.days_names.index(day_name) #update list of names
            del(self.days_names[idx])
            for s in self.substrates_refs:
                del(s.days_refs[idx]) #update list of refs
    
    def display_dynamic(self, hierarchy: dict, days: list, image_type: ImageType, ext=None):

        display_montage_organoids_dynamic(hierarchy, [self], days, image_type, ext)
        
        # #Check substrate exist
        # if bool(set(substrates) & set(self.substrates_names)) == False:
        #     raise ValueError(f"Substrate {"-".join(substrates)} do not exist for experiment {self.name}")

        # #Check if hives are similar both 
        
        # #Determine image dimension
        # filename = image_type.value + self.substrates_refs[0].days_refs[0].hives_refs[0].filename_raw
        # img = self.substrates_refs[0].days_refs[0].hives_refs[0].open_image(rgb=True, filename=filename)
        # dim = img.shape
        # print(dim)
        # del img

        # #Determine the real substrate dimension
        # susbtrates_real = substrates.copy()
        # for s_name in substrates: #Check the substrate names added by user are assigned to the experiment
        #     if s_name not in self.substrates_names:
        #         warnings.warn(f"Substrate {s_name} do not exist.1", UserWarning)
        #         susbtrates_real.remove(s_name)
        # for s in self.substrates_refs: #Check the susbtrate contains at least one day request by user
        #     if s.name in substrates:
        #         if bool(set(days) & set(s.days_names)) == False:
        #             warnings.warn(f"{s.name} do not contain hive days {days}.2", UserWarning)
        #             susbtrates_real.remove(s.name)
        # n_substrates = len(susbtrates_real)
        
        # #Determine the real days dimension
        # days_real = days.copy()
        # missing_days = []
        # for s in self.substrates_refs:
        #     if s.name in substrates:
        #         for d in s.days_refs:
        #             if d.name not in days:
        #                 missing_days.append(d.name)
        # print(missing_days)
        # for n in set(missing_days):
        #     print(n, missing_days.count(n) == len(susbtrates_real))
        #     if missing_days.count(n) == len(susbtrates_real):
        #         warnings.warn(f"Day {n} do not exist for substrate {substrates}.", UserWarning)
        #         days_real.remove(n)
        # n_days = len(days_real)

        # #Determine real hive dimension
        # hive_numbers_real = hive_numbers.copy()
        # missing_hives = []
        # for s in self.substrates_refs:
        #     if s.name in substrates:
        #         for d in s.days_refs:
        #             if d.name in days_real:
        #                 for n in hive_numbers:
        #                     if n not in d.hive_number:
        #                         missing_hives.append(n)
        # for n in set(missing_hives):
        #     if missing_hives.count(n) == len(days_real)*len(susbtrates_real):
        #         warnings.warn(f"Hive {n} pictures do not exist for days {days} in susbtrates {substrates}.", UserWarning)
        #         hive_numbers_real.remove(n)
        # n_hives = len(hive_numbers)

        # print(substrates, days, hive_numbers)
        # print(susbtrates_real, days_real, hive_numbers_real)

        # #Subplot dimension and legend
        # if n_hives < n_days:
        #     nrows = n_hives
        #     ncols = n_days
        #     ttl_subplots = days_real
        #     ylabel = hive_numbers
        # else:
        #     nrows = n_days
        #     ncols = n_hives
        #     ttl_subplots = hive_numbers
        #     ylabel = days_real

    @classmethod
    def _from_csv(cls, csv_filename):
        if os.path.isfile(csv_filename) == False:
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

    parent: Exp #enable autocompletion and will hold Exp instance
    days_refs: list[Day] #enable autocompletion and will hold Day instances

    def __init__(self, name, parent: Exp, days, params):

        self.name = name
        self.parent = parent #enable autocompletion and will hold Substrate instances
        self.days_names = days
        self.params = params
        self.days_refs = []

        for d in days:
            current_param = None
            for row in self.params:
                if row["Substrates"] == self.name and row["Days"] == d:
                    current_param = row
                    break
            if current_param != None:
                day = Day(d, self, current_param)
                self.days_refs.append(day)

    def __repr__(self):
        days = ",".join(self.days_names)
        rep = "Substrate(" + self.name + ", " + self.parent.name + ", " + days + ")"
        return rep
    
    def save_results(self):
        pass

    def display_hives_dynamic(self, hierarchy: dict, days: list, image_type: ImageType, ext=None):
        """"""
        display_montage_organoids_dynamic(hierarchy, [self.parent], days, image_type, ext)

    def crop_hives(self, days: List[str]=None):

        for d in self.days_refs:
            if days is not None:
                if d in days:
                    d.crop_hives()
            else:
                d.crop_hives()
    
    def optimized_crop_hives(self, n_slice: int=15, k_step: int=10, days: List[str]=None):

        for d in self.days_refs:
            if days is not None:
                if d in days:
                    d.optimized_crop_hives(n_slice=n_sclice, k_step=k_step)
            else:
                d.optimized_crop_hives(n_slice=n_sclice, k_step=k_step)

    def threshold_hives(self, display='off', verbose='off', days: List[str]=None):
        pass



        
class Day:

    parent: Substrate #enable autocompletion and will hold the calling Substrate instance
    hives_refs: list[Hive] #enable autocompletion and will hold Day instances

    def __init__(self, name, parent, current_param):

        self.name = name
        self.parent = parent
        self.current_param = current_param
        self.hives_refs = []
        self.hive_number = [] #will hold the static number (ie depending on aquisition parameter)
        self.image_number = [] #will hold the image number 
        self.img_original_dim = None
        self.img_crop_dim = None
        self.img_optimizedshadow_dim = None
        self.img_overlay_dim = None
        self.img_mask_dim = None
        self.path = self.parent.parent.path + path_delimiter + \
            self.parent.name + path_delimiter + \
            self.name + path_delimiter

        #Store filenames if exist, else empty list
        self.filenames_original = self.get_filenames(ImageType.ORIGINAL.value)
        self.filenames_crop = self.get_filenames(ImageType.CROP.value)
        self.filenames_optimized_shadow = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value)
        self.filenames_mask = self.get_filenames(ImageType.MASK.value)
        self.filenames_overlay = self.get_filenames(ImageType.OVERLAY.value)

        if len(self.filenames_original) == 0:
            warnings.warn(f"{self.path} do not contain any usable picture.", UserWarning)
            #return print(f"WARNING: {self.path} do not contain any usable picture.")

        #Verify images have consistent dimensions
        for ty in ImageType:
            if ty is not ImageType.MONTAGE:
                if len(self.get_attribute_name(ty, 'filenames_', '')) > 0:
                    self.check_img_dims(ty)

        #Assign parameters (hive real size and pixel ratio)
        self.hive_height, self.n_expect, self.um2pxl = assign_dim(self.current_param["Design"], \
            self.current_param["Microscope"], self.current_param["Magnification"], self.current_param["Dimension"])
        self.hive_side = HexagonMath.get_side_from_height(self.hive_height) #round(self.hive_height/(2*np.sin(np.pi/3)))
        self.hive_diag = HexagonMath.get_diagonal_from_height(self.hive_height) #2*self.hive_side

        #Create associated hive mask
        center = get_img_center(self.img_original_dim)
        side_pxl = round(self.hive_side*self.um2pxl) #convert hexagon side from µm to pxl
        self.hive_mask = hexagon_outline_ndarray(self.img_original_dim, center, side_pxl, thickness=5, return_type='hexagon')
        #Create associated rescaled hive mask by scale_ratio
        dim_rs = (round(self.img_original_dim[0]/scale_ratio), round(self.img_original_dim[1]/scale_ratio))
        center_rs = get_img_center(dim_rs)
        side_pxl_rs = round(side_pxl/scale_ratio)
        self.hive_mask = hexagon_outline_ndarray(dim_rs, center_rs, side_pxl_rs, thickness=5, return_type='hexagon')

        #Generate Hive Class
        for file in self.filenames_original:
            self.image_number.append(image_number_from_filename(file, 'int'))
            self.hive_number.append(get_hive_number(self.image_number[-1], self.current_param["Aquisition order"]))
            hive = Hive(self.hive_number[-1], self, file)
            self.hives_refs.append(hive)
           
    def __repr__(self):
        rep = "Day(" + self.name + ", " + self.parent.name + ", " + self.parent.parent.name + ", " +  repr(self.current_param) + ")"
        return rep
    
    def get_attribute_name(self, image_type: ImageType, prefixe: str, suffixe: str):
        attr_name = f"{prefixe}{image_type.name.lower()}{suffixe}"
        return getattr(self, attr_name)
    
    def check_img_dims(self, image_type: ImageType):
        """Verify the images have similar dimensions"""
        dims = []
        folder_path = self.parent.parent.path + path_delimiter + self.parent.name + path_delimiter + self.name
        for files in self.get_attribute_name(image_type, 'filenames_', ''):
            with Image.open(folder_path + path_delimiter + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            raise RuntimeError(f"{image_type.name}  images don't have same dimension in {folder_path}")
        elif len(set(dims)) == 1:
            dim_name =f"img_{image_type.name.lower()}_dim"
            setattr(self, dim_name, (dims[0][1], dims[0][0]) )
        else:
            raise RuntimeError(f"{folder_path} issue, no {image_type.name} images found")
    
    def get_filenames(self, image_type: ImageType):
        
        #Initiate output
        filenames = []
        
        #Adjust filename pattern:
        pattern = self.parent.parent.filename_pattern
        pattern = pattern.replace("ExpName", self.parent.parent.name)
        pattern = pattern.replace("Substrate", self.parent.name)
        pattern = pattern.replace("Day", self.name)
        pattern = image_type + pattern
        other_pattern = ImageType.get_attr()
        if image_type == "":
            del other_pattern[0]
        else:
            del other_pattern[0]
            other_pattern.remove(image_type)

        #Search for files
        folder_path = self.parent.parent.path + path_delimiter + self.parent.name + path_delimiter + self.name
        if os.path.isdir(folder_path) == False:
            warnings.warn(f"{folder_path} do not exist.", UserWarning)
        else:
            files = os.listdir(folder_path)
            for f in files:
                if (pattern in f) and (any(ext in f for ext in ImageFormat.get_attr()) and any(ty in f for ty in other_pattern)==False) :
                    filenames.append(f)

        return filenames
    
    def find_rotation_angle(self, scale_ratio, img):
        best_angles = []
        n_hives = len(self.hives_refs)
        for i, hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Computing best angle... {i}/{n_hives}", end="\r")
            best_angles.append(hive.find_hive_rotation_angle(self.hive_mask_rs, int(self.current_param["Angle"]), 5, 0.5, scale_ratio, image=img[i]))
        best_angle = np.nanmedian(best_angles)
        print(f"{self.parent.name}_{self.name}, Computing best angle done: {best_angle}°", end="\r")
        print()
        return best_angle

    def create_montage(self, image_type: ImageType, remove: List[int]=[]):
        # Substrate is definied as follow:
        # 1st column: 3 hives
        # 2nd column: 4 hives
        # 3rd column: 5 hives
        # 4th column: 3 hives
        # 5th column: 4 hives

        #Image acquisition order:
        idx = idx_acquisition_order(self.current_param["Aquisition order"])

        #Open images
        filenames = []
        [filenames.append(image_type.value + f) for f in self.filenames_original]
        #print(filenames)
        img = [None for _ in range(len(self.hives_refs))]
        i=0
        for hive, filename in zip(self.hives_refs, filenames):
            img[i] = hive.open_image(rgb=True,filename=filename)
            i+=1
        
        #Adjust filename pattern:
        pattern = self.parent.parent.filename_pattern
        pattern = pattern.replace("ExpName", self.parent.parent.name)
        pattern = pattern.replace("Substrate", self.parent.name)
        pattern = pattern.replace("Day", self.name)
        pattern = image_type.value + pattern

        #If missing pictures, adding empty ones
        img, filenames = add_empty_img(img, filenames, image_type.value + pattern, 19)
        #img_rs, filenames = add_empty_img(img_rs, filenames, user_inputs["Filenames"], n_expected)

        #Get coordinate for montage
        img_sz = img[0].shape
        img_type = img[0].dtype
        coord, idx_flat = get_hive_coord_montage(img_sz, idx, base='hive')

        #Reconstruct picture
        montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=img_type)
        for i in range(len(img)):
            montage[coord[i][1][0]:coord[i][1][1], coord[i][0][0]:coord[i][0][1], :] = img[idx_flat[i]]

        #Save montage
        if image_type is ImageType.CROP:
            out_path = self.parent.parent.path + path_delimiter + \
            "Montage_" + image_type.value + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + ".jpg"
        else:
            out_path = self.path + path_delimiter + \
            "Montage_" + image_type.value + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + ".jpg"
        imsave(out_path, montage)

        print(f"Montage image saved: {out_path}")

        return montage

    def save_results(self):
        pass

    def crop_hives(self):
        
        img = [None for _ in range(len(self.filenames_original))]
        img_rs = [None for _ in range(len(self.filenames_original))]
        n_hives = len(self.hives_refs)

        #Open images
        for i,hive in enumerate(self.hives_refs):
            img[i] = hive.open_image(rgb=True)
            img_rs[i] = resize(rgb2gray(img[i]), (round(img[i].shape[0]/scale_ratio), round(img[i].shape[1]/scale_ratio))) #rescale image

        #Compute rotation angle
        best_angle = self.find_rotation_angle(scale_ratio, img_rs)

        #Crop & rotate image
        for i,hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Cropping pictures... {i}/{n_hives}", end="\r")
            hive.crop_hive(scale_ratio, best_angle, save=True, image=img[i], image_rs=img_rs[i])

        #Update Crop filenames
        self.filenames_crop = self.get_filenames(ImageType.CROP.value)

        #Verify crop pictures have same dimension
        folder_path = self.parent.parent.path + path_delimiter + self.parent.name + path_delimiter + self.name
        dims = []
        for files in self.filenames_crop:
            with Image.open(folder_path + path_delimiter + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            raise RuntimeError(f"Cropped images don't have same dimension in {folder_path}")
        self.img_crop_dim = (dims[0][1], dims[0][0])

        print(f"{self.parent.name}_{self.name}, Croping pictures done: {self.img_crop_dim}", end="\r")
        print()

    def optimized_crop_hives(self, n_slice: int=15, k_step: int=10):

        n_hives = len(self.hives_refs)
        hives_mask_side = [None] * n_slice

        for j,k in enumerate(range(k_step, n_slice*k_step+k_step, k_step)):
            side_k = round((self.hive_height*self.um2pxl - 2*k)/(2*np.sin(np.pi/3)))
            hives_mask_side[j] = hexagon_outline_ndarray(self.img_crop_dim, [round(self.img_crop_dim[0]/2), round(self.img_crop_dim[1]/2)], side_k, thickness=k_step, return_type='side')
        #Otimize crop and save it
        for j,hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Refine cropping... {j}/{n_hives}", end="\r")
            hive.optimized_crop_hive(hives_mask_side, k_step)

        #Update filenames
        self.filenames_optimized_shadow = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value)
        print(self.filenames_optimized_shadow)

        print(f"{self.parent.name}_{self.name}, Refine cropping done.", end="\r")
        print()

    def threshold_hives(self, opticrop_image=None, display='off', verbose='off'):

        """For entire session, determine each threshold to apply on each picture to select organoids.
        Thresholds are stored in a list of tuptle: each element gives the threshold of one picture and each element is a tuptle of 2 numpy floats."""
        
        n_hives = len(self.filenames_optimized_shadow)

        #Open pictures
        if opticrop_image is None:
            opticrop_img = []
            for hive in self.hives_refs:
                filename = ImageType.OPTIMIZED_SHADOW.value + hive.filename_raw
                if '.jpg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                elif '.jpeg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                opticrop_img.append(hive.open_image(filename=filename, rgb=True))
        else:
            opticrop_img = opticrop_image

        #Create figure is display == 'on'
        if display == 'on':
            n_r, n_c = get_compact_grid(n_hives) #adequate subplot grid
            fig, ax = plt.subplots(nrows=n_r, ncols=n_c, sharex=True, sharey=True, 
                                   num=f"Threshold for {self.parent.parent.name}_{self.parent.name}_{self.name}") #create subplots
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
        thresholds = []
        for j,hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Determining threshold... {j}/{n_hives}", end="\r")
            if display == 'on':
                current_ax =  ax[idx_ax[j]]
            else:
                current_ax = None
            thresholds.append(hive.custom_threshold_hive(opticrop_img=opticrop_img[j], fig1=fig, ax1=current_ax, title=f"#{hive.hive_number}", verbose=verbose))

        print(f"{self.parent.name}_{self.name}, Determining threshold done.    ", end="\r")
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

    def generate_mask_hives(self, opticrop_image=None, display='off', verbose='off'):
        
        n_hives = len(self.filenames_optimized_shadow)
        
        #Open optimized crop pictures
        if opticrop_image is None:
            opticrop_img = []
            for hive in self.hives_refs:
                filename = ImageType.OPTIMIZED_SHADOW.value + hive.filename_raw
                if '.jpg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                elif '.jpeg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                opticrop_img.append(hive.open_image(filename=filename, rgb=True))
        else:
            opticrop_img = opticrop_image

        #Create subplot
        if display == 'on':
            #1st dim: images
            #2nd dim: original, custom_threshold, binary operation, filtered, overlay?
            fig, ax = plt.subplots(ncols=n_hives, nrows=4, num=f"Segmentation Steps - {self.parent.parent.name}_{self.parent.name}_{self.name}")
            ax[0, 0].set_ylabel("Original")
            ax[1, 0].set_ylabel("Threshold")
            ax[2, 0].set_ylabel("Binary operations")
            ax[3, 0].set_ylabel("Mask")
        else:
            fig=None
            ax=None

        #Generate mask
        for j,hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Generating organoid mask... {j}/{n_hives}", end="\r")
            if display == 'on':
                current_ax_idx =  j
            else:
                current_ax_idx = None
            hive.generate_organoid_mask(opticrop_img=opticrop_img[j], fig=fig, ax=ax, ax_idx=current_ax_idx, verbose=verbose)

        print(f"{self.parent.name}_{self.name}, Generating organoid mask done.    ", end="\r")
        print()

        #Display
        if display == 'on':
            plt.show()

    def save_results(self, masks=None, opticrop_image=None, manual_discard=True, ResultFile="Results.csv"):
        
        n_hives = len(self.filenames_mask)

        #Open masks
        if mask is None:
            mask = []
            for hive in self.hives_refs:
                filename = ImageType.MASK.value + hive.filename_raw
                if '.jpg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                elif '.jpeg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                mask.append(hive.open_image(filename=filename, rgb=True))
        else:
            mask = opticrop_image

        #Open OptimizedShadow images
        if opticrop_image is None:
            opticrop_img = []
            for hive in self.hives_refs:
                filename = ImageType.OPTIMIZED_SHADOW.value + hive.filename_raw
                if '.jpg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                elif '.jpeg' in filename:
                    filename = filename.replace('.jpg', '.tif')
                opticrop_img.append(hive.open_image(filename=filename, rgb=True))
        else:
            opticrop_img = opticrop_image

        #Open csv file to store segmentation results
        f = open(self.path + path_delimiter + "Results.csv", 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        #For each hive, get data & write in in result file
        data = []
        for h, hive in enumerate(self.hives_refs):
            data.append(hive.get_organoid_data(image=opticrop_image[h], mask_image=mask[h]))
            writer.writerows(data[s][d][i])
        f.close()

        #Perform manual discard
        if manual_discard == True:
            overlay_montage = self.create_montage(ImageType.OVERLAY)

    def manual_discard(self, overlay=None, newResultFile="Updated", ResultFile="Results.csv"):
        pass
        
        # #Open overlay image
        # if overlay is None:
        #     overlay_path = self.path + path_delimiter +  "Montage_" + ImageType.OVERLAY + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + ".jpg"
        #     overlay = np.array(imread(overlay_path))
        # else:
        #     overlay = overlay

        # #Read and store data
        # f = open(self.path + path_delimiter + newResultFile, 'r', newline='')
        # reader = csv.Dict(f, fieldnames=headers)

        # #Get index for aquisition order
        # idx = idx_acquisition_order(self.current_param["Aquisition order"])
        # #Get hive coordinates
        # coord_hives, idx_flatten = get_hive_coord_montage(overlay.shape, idx, base='montage')
        # #Reorganize coordinates into list of dictionnary to match data organisation
        # coords = []
        # for i in range(len(data)):
        #     for j in range(len(data)):
        #         coords.append({"Index": data[i][j]["Index"], \
        #                        "Hive number": data[i][j]["Hive number"], \
        #                        "Hive coord": coord_hives[idx_flatten.index(data[[i][j]["Hive number"])],
        #                        "bbox": (data[i][j]["bbox_ymin"]/6, data[i][j]["bbox_xmin"]/6, data[i][j]["bbox_ymax"]/6, data[i][j]["bbox_xmax"]/6) #coordinates of resize image, division factor of 6 to have the rescale one
        #                        })     
        
        # #Create figure
        # fig3, ax3 = plt.subplots()
        # ax3.imshow(overlay)
        # discarded_organoids = [] #will hold the discarded hives
        # handler_on_click = partial(on_click_discard_undiscard_organoid, coords=coords, discarded_organoids=discarded_organoids) #to enable passing variable to plt.connect functions
        # fig3.canvas.mpl_connect('button_press_event', handler_on_click) #callbaks to discard/undiscard hives when selecting a hive on the picture
        # plt.show()

        # #Once picture is closed, update data & update results file
        # f = open(self.path + path_delimiter + newResultFile, 'w+', newline='')
        # writer = csv.DictWriter(f, fieldnames=headers)
        # writer.writeheader()
        # if len(discarded_organoids) > 0:
        #     for i in range(len(data[s][d])): #loop over each picture (ie hive)
        #         #print(data[s][d][i], type(data[s][d][i]), len(data[s][d][i]))
        #         if len(data[s][d][i]) > 0:
        #             for j in range(len(data[s][d][i])): #loop over each object from the picture (ie hive)
        #                 if data[s][d][i][j]["Index"] in discarded_organoids:
        #                     data[s][d][i][j]["Manually Discarded"] = True
        #                 else:
        #                     data[s][d][i][j]["Manually Discarded"] = False
        #             writer.writerows(data[s][d][i])
        # else: #simply copy the data in UpdatedResults.csv
        #     for i in range(len(data[s][d])):
        #         writer.writerows(data[s][d][i])
        # f.close()
        # print(f"The discarded hives are: {discarded_organoids}")
             
    def extract_metrics_from_mask_and_original(self, metrics: List[str]=[]):
        pass

class Hive:

    parent: Day #enable autocompletion and will hold the calling Day instance

    def __init__(self, hive_number, parent, filename_raw):

        self.hive_number = hive_number
        self.parent = parent
        self.filename_raw = filename_raw
        self.name = self.parent.parent.parent.name + "_" + self.parent.parent.name + "_" + self.parent.name + "_" + str(self.hive_number) + ": " + self.filename_raw
        self.results = []
        self.crop_coord = None
        self.threshold = (0, 1)

    def open_image(self, rgb=False, img_type=None, filename=None):

        if filename is None:
            file_name = self.filename_raw
        else:
            file_name = filename
        filepath = self.parent.parent.parent.path + path_delimiter + self.parent.parent.name + path_delimiter + self.parent.name + path_delimiter + file_name
        if os.path.isfile(filepath) == False:
            raise ValueError(f"{filepath} not found. {file_name} couldn't be opened.")
        
        img = np.array(imread(filepath))
        if rgb == False:
            img = rgb2gray(img)

        if img_type != None:
            img = img.astype(img_type)

        return img
    
    def find_hive_rotation_angle(self, hive_mask, angle_user, angle_range, angle_step, scale_ratio, image=None):
        """returns rotation angle of the hive. Rotation angle defined as the best angle in the range 
        [angle_user-angle_range; angle_user+angle_range]. Best angle defined as convolution ouput of 
        picture with rotated hexagon containing maximum value.
        Note: if image is None, the raw image of the hive is opened. """

        #Open image
        if image is None:
            img = self.open_image()
            img = resize(img, (round(img.shape[0]/scale_ratio), round(img.shape[1]/scale_ratio)))
        else:
            img = image
        
        angles = np.arange(angle_user-angle_range, angle_user+angle_range, angle_step)
        conv_max_value = [] #will hold list on maximum value of each convolution output

        for a in angles:
            conv = fftconvolve(img, rotate(hive_mask, a), mode='same') #compute convolution of hive image with a rotation of hive mask
            conv_max_value.append(conv.flatten().max()) 

        return angles[conv_max_value.index(max(conv_max_value))]
       
    def crop_hive(self, scale_ratio, best_angle, save=True, image=None, image_rs=None):
        """Return & save (if save is True) cropped image and update crop coordinates self.crop_coord = (ymin, ymax, xmin, xmax)"""

        if image is None:
            img = self.open_image(rgb=True) #open image
        else:
            img = image
        img_type = img.dtype

        if image_rs is None:
            img_rs = rgb2gray(img)
            img_rs = resize(img_rs, (round(img.shape[0]/scale_ratio), round(img.shape[1]/scale_ratio))) #rescale image
        else:
            img_rs = image_rs

        (y_c, x_c) = self.get_hive_barycenter(best_angle, self.parent.hive_mask_rs, scale_ratio, image=img_rs) #compute hive barycenter

        #rotate image
        img = rotate(img, best_angle, preserve_range=True, resize=False) #weird, before needed to write resize=True and now need o write False
        img = img.astype(img_type)
        img = adjust_dimension_after_rotation(img, [round(self.parent.hive_height*self.parent.um2pxl), round(self.parent.hive_diag*self.parent.um2pxl)], img_type) #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)

        #Compute x and y values where crop should occure:
        y_cut = self.cut_value_for_crop(y_c, self.parent.hive_height*self.parent.um2pxl/2, img.shape[0])
        x_cut = self.cut_value_for_crop(x_c, round(self.parent.hive_diag*self.parent.um2pxl/2), img.shape[1]) #round here to get consistent dimension over pictures

        # Fill the hexagon mask with 1 value
        hive = flood_fill(self.parent.hive_mask, (y_c, x_c), 1) #fill the hexagon
        hive = hive.astype(int)
        #Force inner part to be set to at 1
        if hive[y_c, x_c] == 0:
            hive = np.invert(hive)

        #Crop the picture
        img[hive == 0] = 0 #assign all values outside hive to 0
        img = img[y_cut[0]:y_cut[1], x_cut[0]:x_cut[1], :] #crop around h and diag

        #Update crop_coord attribute
        self.crop_coord = (y_cut[0], y_cut[1], x_cut[0], x_cut[1])

        #Save the cropped picture
        if save == True:
            out_path = self.parent.path + path_delimiter + \
                ImageType.CROP.value + self.filename_raw
            if '.jpg' in out_path:
                out_path = out_path.replace('.jpg', '.tif')
            elif '.jpeg' in out_path:
                out_path = out_path.replace('.jpg', '.tif')
            imsave(out_path, img)

        #return img

    def optimized_crop_hive(self, hives_mask_side, k_step, crop_img=None):

        #Open image in grayscale
        if crop_img is None:
            filename = ImageType.CROP.value + self.filename_raw
            if '.jpg' in filename:
                filename = filename.replace('.jpg', '.tif')
            elif '.jpeg' in filename:
                filename = filename.replace('.jpg', '.tif')
            img = self.open_image(rgb=True, filename=filename)
        else: 
            img = crop_img
        img_gray = rgb2gray(img)

        #Remove artefacts/noise
        r_blur=5*self.parent.um2pxl #to blur unique cells entierly (considered cells about 10µm diameter)
        img_gray = gaussian_filter(img_gray, r_blur)

        #Variables
        img_shape = img_gray.shape #shape of the picture
        x_c = round(img_shape[1]/2) #center of the picture along x axis
        y_c = round(img_shape[0]/2) #center of the picture along y axis
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

            #Compute the means and deltaQ variations
            means_derivative = []
            for k in range(len(means)-1):
                means_derivative.append((float(means[k+1]-means[k])/2))

            #Determine the slice at which there is no more shadow,
            #ie when there is a high change in pixel intensity,
            #ie at 80% of the length between the maximum of the derivative and the 1st following 0
            mean_derivative_max = means_derivative.index(max(means_derivative))
            mean_derivative_1st_0 = means_derivative[-1]
            for elt in range(mean_derivative_max, len(means_derivative)):
                if means_derivative[elt] < 0:
                    mean_derivative_1st_0 = elt
                    break
            cut_idx[s_k] = round((mean_derivative_1st_0 - mean_derivative_max)*0.8 + mean_derivative_max)

            #Store the coordinates of the fine tuned vertices of the side s_k
            out = hexagon_outline_ndarray(img_shape, [y_c, x_c], round((self.parent.hive_height*self.parent.um2pxl - 2*cut_idx[s_k]*k_step)/(2*np.sin(np.pi/3))), thickness=5, return_type='coord-side')
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

        #Save the Optimized cropped picture
        img[hive_mask == 0] = 0
        out_path = self.parent.path + path_delimiter + "OptimizedShadow_" + self.filename_raw
        if '.jpg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        elif '.jpeg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        imsave(out_path, img)

    def custom_threshold_hive(self, opticrop_img=None, fig1=None, ax1=None, title=None, verbose='off'):

        if fig1 is None and ax1 is None:
            fig, ax = plt.subplots()
        else:
            fig = fig1
            ax = ax1
        
        #Open image in grayscale
        if opticrop_img is None:
            filename = ImageType.OPTIMIZED_SHADOW.value + self.filename_raw
            if '.jpg' in filename:
                filename = filename.replace('.jpg', '.tif')
            elif '.jpeg' in filename:
                filename = filename.replace('.jpg', '.tif')
            img = self.open_image(rgb=True, filename=filename)
        else: 
            img = opticrop_img
        img_gray = rgb2gray(img)

        #Remove artefacts
        r_blur=5*self.parent.um2pxl #to blur unique cells entierly (considered cells about 10µm diameter)
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

        #Find the peaks
        peaks = find_peaks(data[0], distance=10, height=max(data[0])*0.005)
        if len(peaks[0]) == 0 or len(peaks[0]) == 1:
            return [0, 1], "NA", "NA"

        #Split the histogram in two groups
        otsu = round(threshold_otsu(img_gray) * pxl_depth)

        #Get the peak info of 1st half of histogram
        peaks_low = np.where(peaks[0] < otsu)
        if len(peaks_low[0]) != 0:
            idx_low = np.where(peaks[1]["peak_heights"] == max(peaks[1]["peak_heights"][peaks_low]))[0][0]
            peak_low = [peaks[0][idx_low]/pxl_depth , peaks[1]["peak_heights"][idx_low]]
            ax.scatter(peak_low[0], peak_low[1], c='r', marker='x')
        else:
            return [0, 1], "NA", "NA"

        #Get the peak info of 2nd half of histogram
        peaks_high = np.where(peaks[0] >= otsu)
        if len(peaks_high[0]) != 0:
            idx_high = np.where(peaks[1]["peak_heights"] == max(peaks[1]["peak_heights"][peaks_high]))[0][0]
            peak_high = [peaks[0][idx_high]/pxl_depth , peaks[1]["peak_heights"][idx_high]]
            ax.scatter(peak_high[0], peak_high[1], c='g', marker='x')
        else:
            return [0, 1], "NA", "NA"

        if peaks[1]["peak_heights"][idx_high] > peaks[1]["peak_heights"][idx_low]:
            #First fit high peak
            guess_high = [
                peak_high[0], #center of gaussian
                peak_high[1], #amplitude
                1 - peak_high[0] #FWHM
            ]
            try:
                popt_high, pcov_high = curve_fit(gauss_fit, x, data[0], p0=guess_high)
            except:
                popt_high = [0, 0, 0]
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
            fit_high = gauss_fit(x, *popt_high)

            #Remove data from high peak to enable correct fit of low part
            data[0][otsu:-1] = 0

            #Fit low peak
            guess_low = [
                peak_low[0], #center of gaussian
                peak_low[1], #amplitude
                peak_low[0] #FWHM
            ]
            try:
                popt_low, pcov_low = curve_fit(gauss_fit, x, data[0], p0=guess_low)
            except:
                popt_low = [0, 0, 0]
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
            fit_low = gauss_fit(x, *popt_low)

        elif peaks[1]["peak_heights"][idx_high] < peaks[1]["peak_heights"][idx_low]:
            #First fit low peak
            guess_low = [
                peak_low[0], #center of gaussian
                peak_low[1], #amplitude
                peak_low[0] #FWHM
            ]
            try:
                popt_low, pcov_low = curve_fit(gauss_fit, x, data[0], p0=guess_low)
            except:
                popt_low = [0, 0, 0]
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
            fit_low = gauss_fit(x, *popt_low)

            #Remove data from low peak to enable correct fit of high part
            data[0][0:otsu] = 0

            #Fit high peak
            guess_high = [
                peak_high[0], #center of gaussian
                peak_high[1], #amplitude
                1 - peak_high[0] #FWHM
                ]
            try:
                popt_high, pcov_high = curve_fit(gauss_fit, x, data[0], p0=guess_high)
            except:
                popt_high = [0, 0, 0]
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
            fit_high = gauss_fit(x, *popt_high)

        #Set threshold for high pixel values depending on the spread of the fit ie when the gaussian encouters the center of the low intensity gaussian
        if popt_high[0] - popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] #center
            high_cut_type = "\u03BC_high"
        elif popt_high[0] - 2*popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] - popt_high[2] #1 sigma
            high_cut_type = "\u03BC_high - \u03C3"
        elif popt_high[0] - 2*popt_high[2] <= popt_low[0]:
            pxl_max = popt_high[0] - 2*popt_high[2] #2 sigma
            high_cut_type = "\u03BC_high - 2\u03C3"
        else:
            pxl_max = popt_high[0] - 3*popt_high[2] #3 sigma
            high_cut_type = "\u03BC_high - 3\u03C3"

        #Set threshold for low pixel value
        if popt_low[0] < 0:
            pxl_min = 0
            low_cut_type = "0"
        elif popt_low[0] - popt_low[2] < 0:
            pxl_min = popt_low[0]
            low_cut_type = "\u03BC_low"
        else:
            pxl_min = popt_low[0] - popt_low[2]
            low_cut_type = "\u03BC_low - \u03C3"

        #Verify there is no aberation on threshold values
        if pxl_min < 0:
            pxl_min = 0
        if pxl_max <=0:
            print("pxl_max < 0 --> Threshold not found.")
            pxl_min = 0
            pxl_max = 1
        if pxl_max - pxl_min <= 0:
            print("Threshold not found.")
            pxl_min = 0
            pxl_max = 1

        x_cut = [pxl_min, pxl_max]

        #Prints
        if verbose == 'on':
            print(f"{self.name}, Determination of threshold values done:")
            print("\tOtsu 2 groups thresh =", otsu)
            print(f"\tLow pixel intensity peak (x,y): {peak_low}; \tGaussian fit Params: {popt_low}; \tMin threshold defined as {low_cut_type}")
            print(f"\tHigh pixel intensity peak (x,y): {peak_high}; \tGaussian fit Params: {popt_high}; \tMin threshold defined as {high_cut_type}")
            print("\tThreshold value (in range [ 0|0 ; 1|255 ]: ", end=" ")
            [print(f"{elt}|{elt*pxl_depth}", end=" ") for elt in x_cut]
            print()

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
        self.threshold = x_cut

        return x_cut#, low_cut_type, high_cut_type
    
    def generate_organoid_mask(self, opticrop_img=None, fig=None, ax=None, ax_idx=None, verbose='off'):

        mesh_pxl = round(60*self.parent.um2pxl/2)
        
        #Open image in grayscale
        if opticrop_img is None:
            filename = ImageType.OPTIMIZED_SHADOW.value + self.filename_raw
            if '.jpg' in filename:
                filename = filename.replace('.jpg', '.tif')
            elif '.jpeg' in filename:
                filename = filename.replace('.jpg', '.tif')
            img = self.open_image(rgb=True, filename=filename)
        else: 
            img = opticrop_img
        img_gray = rgb2gray(img)

        #Remove artefacts
        r_blur=5*self.parent.um2pxl #to blur unique cells entierly (considered cells about 10µm diameter)
        img_blur = gaussian_filter(img_gray, r_blur)

        #Apply threshold
        img_thresh = (img_blur > self.threshold[0]) & (img_gray <= self.threshold[1])

        #Close areas
        img_bin = binary_erosion(img_thresh, footprint=ellipse(mesh_pxl, mesh_pxl)) #to get rid of the segmented mesh
        img_bin = area_closing(img_bin)
        img_bin = binary_dilation(img_bin, footprint=ellipse(mesh_pxl, mesh_pxl))

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
            if (rg.eccentricity > 0.88) or (rg.area_filled <= 30000) or (rg.area_filled >= 2000000): #(rg.area_filled <= 10000*self.parent.um2pxl) or (rg.area_filled >= 1500000*self.parent.um2pxl):
                labs.append(rg.label)
        #Remove them
        for k in sorted(labs, reverse=True):
            del(regions[k-1])
            labels[(labels == k)] = 0

        if verbose == 'on':
            print()
            print(f"Image {self.filename_raw}: {np.unique(labels)} --> {len(np.unique(labels)) - 1} organoids found.")
            print()

        #Fill the holes in accurate regions
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

        #Save mask
        out_path = self.parent.path + path_delimiter +  "Mask_" + self.filename_raw
        imsave(out_path, organoid_mask)

        #Save overlay of raw picture + organoid mask (in red)
        overlay = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        overlay[:,:,0] = img[:,:,0] + organoid_mask*0.2 #R
        overlay[:,:,1] = img[:,:,1] #G
        overlay[:,:,2] = img[:,:,2] #B
        out_path = self.parent.path + path_delimiter +  "Overlay_" + self.filename_raw
        imsave(out_path, overlay)
        
    def get_organoid_data(self, image=None, mask_image=None, properties=OrganoidProperties.props.value, metadata=OrganoidProperties.metadata.value):
        """Return organoid data & metadata as a list of dictionnary.
        Each element is one organoid and each key is a propertie from skimage.regionprop or a metadata."""
        
        #Open mask
        if mask_image is None:
            mask_img = self.open_image(filename="Mask_" + self.filename_raw)
        else: 
            mask_img = mask_image
        
        #Open original image
        if image is None:
            img = self.open_image(filename=self.filename_raw)
        else: 
            img = image

        #Ectract info using region props
        labels = label(mask_img)
        regions = regionprops(labels, img)
        data = []
        organoid_count = 0
        for rg in regions:
            rg_props = get_all_attribute_names(rg)
            data.append({})
            [data[-1].update({head: rg[head]}) for head in props if head in rg_props] #from regionprops
            data[-1].update({"centroidX": float(rg.centroid[0]), "centroidY": float(rg.centroid[1]), 
                             "centroid_localX": rg.centroid_local[0], "centroid_localY": rg.centroid_local[1]}) #from regionprops
            data[-1].update({"bbox_ymin": float(rg.bbox[0]), "bbox_xmin": float(rg.bbox[1]), 
                             "bbox_ymax": rg.bbox[2], "bbox_xmax": rg.bbox[3]}) #from regionprops
            data[-1].update({"Index": organoid_count, "Day": self.parent.name, "Substrate": self.parent.parent.name,
                             "Image Name": self.filename_raw, "Hive number": image_number_from_filename(self.filename_raw, format='int'),
                             "Manually Discarded": False, "Analysis Date": datetime.now(), "Analysis Version": VERSION}) #metadata
            organoid_count += 1 #incremente organoid count
        
        return data

    def cut_value_for_crop(self, barycenter, crop_length, img_length):
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
    
    def get_hive_barycenter(self, best_angle, hive_mask, scale_ratio, image=None):

        #Open image
        if image is None:
            img = self.open_image()
            img = resize(img, (round(img.shape[0]/scale_ratio), round(img.shape[1]/scale_ratio)))
        else:
            img = image

        if sum(sum(img)) == 0:
            return (0, 0)
        
        #Pre-process of the picture
        blur = gaussian_filter(img, 5/scale_ratio) #blur pictire to smooth edges
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
        y_c = round(self.parent.img_original_dim[0]*y_c/conv.shape[0])
        x_c = round(self.parent.img_original_dim[1]*x_c/conv.shape[1])

        return (y_c, x_c)


exp7 = Exp._from_csv("Book1.csv")
exp1 = Exp._from_csv("Book2.csv")
hierarchy = {
    "Exp007-B": {
        "H2M150": [0, 9, 18, 20],
        "H2M100": [0, 5, 18]
    },
    "Exp001-B": {
        "CondB": [0, 9, 15, 18],
        "CondC": [0, 9, 15, 18]
    },
    "Prout": {
        "prout": [0, 9, 15, 18]
    }
}
exp7.substrates_refs[0].days_refs[0]
# exp7.display_hierarchy()
# exp1.display_hierarchy()
display_montage_organoids_dynamic(hierarchy, [exp7, exp1], ["D0", "D1", "D4", "D6", "D11", "D19"], ImageType.ORIGINAL)
# exp7.display_dynamic(hierarchy, ["D0", "D1", "D4", "D6", "D11", "D19"], ImageType.ORIGINAL)
# exp7.substrates_refs[0].display_hives_dynamic(hierarchy, ["D0", "D1", "D4", "D6", "D11", "D19"], ImageType.ORIGINAL)


# # exp7.substrates_refs[1].days_refs[5].crop_hives()
# # exp7.substrates_refs[1].days_refs[5].optimized_crop_hives()
# # thresh = exp7.substrates_refs[1].days_refs[5].threshold_hives(display='off', verbose='off')
# # exp7.substrates_refs[1].days_refs[5].generate_mask_hives(display='on', verbose='off')
# #print("Image number: ", exp7.substrates_refs[0].days_refs[0].image_number, "Filename: ", exp7.substrates_refs[0].days_refs[0].filenames_original)
# # print(exp7.substrates_refs[0].days_names)
# # exp7.substrates_refs[1].display_hives_dynamic([0, 7, 9, 18, 20], ["D4", "D8"], ImageType.OPTIMIZED_SHADOW)
# exp7.display_dynamic([0, 9, 18, 20], ["D0", "D3", "D4", "D8", "D9"], ["H2M100", "H2M150", "prout"], ImageType.ORIGINAL)        



