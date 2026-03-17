from __future__ import annotations  # recommended by chatgpt
import os
from datetime import datetime
import warnings
import itertools
import csv
import numpy as np
from functools import partial
from typing import List, Tuple
from enum import Enum
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
from scipy.stats import median_abs_deviation
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backend_bases import MouseButton
import re
from math import floor, sqrt, gcd, ceil
import json

"""
Global Image Processing Pipeline
================================

Hierarchy
---------
Exp → Substrate → Day → Hive

Execution Principle
-------------------
- Exp and Substrate are orchestration layers.
- Day is the coordination and aggregation layer.
- Hive is the execution layer (single slice processing).
- Data aggregation occurs at Day level.
- Shared geometric resources are defined once at Exp level.
- Processing is slice-based, but slices are not objects.

Architectural Principle
-------------------
Exp:
    Defines geometry + experiment structure
Substrate:
    Loops over days
Day:
    Loops over slices and hives
    Aggregates results
    Determines final rotation
    Creates montages
    Saves CSV
Hive:
    Executes image processing
    Saves processed images

---------------------------------------------------------------------
What is a "Slice"?
------------------

A slice is an index tuple describing one specific image acquisition:

    slices = (site_idx, time_idx, z_idx, channel_name)

It defines a single image inside a hive stack.

Example:
    (0, 5, 3, ChannelNames.BF)

Means:
    - site 0
    - timepoint 5
    - z-stack index 3
    - brightfield channel

Slices are:
    • Not classes
    • Pure indexing tuples
    • Used only at Day and Hive level

---------------------------------------------------------------------

1. Exp (Experiment Level)
-------------------------

Role:
- Top-level orchestrator.
- Initializes Substrate instances.
- Creates and stores shared geometric resources.
- Controls subset selection via `hierarchy`.

Key Responsibilities:
- Create Substrates
- Manage global parameters
- Initialize gemoetry backbone

Shared Resource Initialization:
- create_hexagon_masks()
    → Computes acquisition-dependent masks
    → Stores them in self.hexagon
    → Used later by Day and Hive

Data Flow:
    self.hexagon
        ↓ (via parent references)
    Substrate → Day → Hive

---------------------------------------------------------------------

2. Substrate (Condition Level)
------------------------------

Role:
- Represents one experimental condition.
- Creates Day instances.
- Loops over selected days.
- Delegates execution to Day.

No heavy computation is done here.
Acts purely as an orchestration layer.

---------------------------------------------------------------------

3. Day (Acquisition Day Level)
------------------------------

Role:
- Manages hives for one substrate on one day.
- Organizes filenames and slice indexing.
- Loops over slices and hives.
- Aggregates results.
- Generates monatge image for visualization of substrate
at one day.
- Computes final rotation angle

Uses:
- Hexagon masks stored in Exp
- Acquisition parameters
- Threshold values

Day-level Responsibilities
--------------------------

1. Slice looping
2. Calling Hive methods
3. Collecting rotation angles
4. Computing final rotation angle:
       median(all hive rotation angles)
5. Aggregating organoid measurements
6. Writing result CSV
7. Creating montage images

Data Aggregation
----------------

Aggregation occurs here:
- Organoid properties are collected per hive
- Merged into a single DataFrame
- One CSV file per substrate-day combination.

Even if called from Exp, aggregation remains at Day level.

---------------------------------------------------------------------

4. Hive (Execution Level)
-------------------------

Role:
- Operates on a single hive.
- Operates on a single slice.
- Performs all image processing steps.

Core Operations:
1. Image opening
2. Rotation estimation (via convolution)
3. Cropping
4. Optimized cropping
5. Thresholding
6. Organoid segmentation
7. Barycenter computation
8. Mask + overlay saving

---------------------------------------------------------------------

Processing Flow
---------------

Exp
 ├── Global initialization: substrate, day, aquisition parameters
 |              ↓
 ├── Geometry initialization: hive masks
 │       
 └── Substrate
      └── Day
        ├── Hive:
        │    ├── Hive.open_image()
        │    ├── Hive.find_hive_rotation_angle() ─────────────────┐
        │    ├── Hive.get_hive_barycenter()                       │
        │    ├── Hive.crop_hive() → autosaves Crop images         │                        
        │    ├── Hive.optimized_crop_hive() → autosaves           │
        │    │   Optimized_Shadow images                          │          
        │    ├── Hive.otsu() / custom_threshold()                 │       
        │    ├── Hive.generate_organoid_mask() → autosaves        │
        │    │   Mask and Overlay images                          │               
        │    └── Hive.save_organoid_data() ────────────────────┐  │        
        │                                                      │  │                                                  
        ├── Creates montage                                    │  │                                    
        ├── Aggregates organoid data → autosaves               │  │
        │   Result file                          <─────────────┘  │
        ├── Manual Discard of organoid → autosaves Updated        │
        │    result file & images with discarded organoids        │
        └── Compute final rotation angle         <────────────────┘               
 
---------------------------------------------------------------------

Key Architectural Principle
---------------------------

Exp creates:
    • Structure
    • Shared geometric masks
    • Global configuration

Substrate loops over:
    • Days

Day loops over:
    • Hives
    • Slices

Hive executes:
    • Image processing
    • Segmentation
    • Measurement

---------------------------------------------------------------------

Important Design Feature
------------------------

Hexagon masks are:
    • Acquisition-dependent
    • Precomputed once at Exp level
    • Reused across all lower levels
    • Accessed through parent references

This avoids recomputation and guarantees
consistent geometric definitions across the experiment.

---------------------------------------------------------------------

Example
------------------------

Here is an example of full pipeline use. You may download and use the
experiments images and Input_csv file to run the pipeline.

In this experiment, 2 substrates, named Old-1 and Old-2, have been imaged 1,
4 and 6 days after cell seeding. All images have been captured using
Leica microscope with a magnification of x5 in BF only and the image
dimension is 1944x2592 pxl² (height,width) for all imaging sessions.
Each image is named `<ExpName>_<SubstrateName>_<DayName>_<ImageNumber>.tiff`.

The folder architecure of the images is as pictured below:
<put image link>

The csv file containing user inputs on experiment looks like this:
<put image link>

```Note: the substrate names entered in Substrates column of csv file should:
- be a substrate folder within Experiment path (`Path` field from csv)
- be the SubstrateName in the images name
In the same way, the substrate names entered in Days column of csv file should:
- be actual day folders within associated Substrate folder
- be the DayName in the images name
```

0. Initialize class instances
--
Use the csz file containing user inputs to create Exp, Substrate, Day and Hive instances of your experiment.
>>> exp = Exp("Demo\\Demo.csv")
>>> exp.display_hierarchy(compact=True)
```
Exp009-B:
        Old-1
                D1: {'Substrates': 'Old-1', 'Days': 'D1', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '2', 'Aquisition order': '0', 'Microscope': 'Leica', 'Magnification': 'x5'}
                D4: {'Substrates': 'Old-1', 'Days': 'D4', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '0', 'Aquisition order': '11-flip', 'Microscope': 'Leica', 'Magnification': 'x5'}
                D6: {'Substrates': 'Old-1', 'Days': 'D6', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '0', 'Aquisition order': '11-flip', 'Microscope': 'Leica', 'Magnification': 'x5'}
        Old-2
                D1: {'Substrates': 'Old-2', 'Days': 'D1', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '2', 'Aquisition order': '0', 'Microscope': 'Leica', 'Magnification': 'x5'}
                D4: {'Substrates': 'Old-2', 'Days': 'D4', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '0', 'Aquisition order': '2-flip', 'Microscope': 'Leica', 'Magnification': 'x5'}
                D6: {'Substrates': 'Old-2', 'Days': 'D6', 'Design': 'H2M100', 'Dimension': '1944x2592', 'Angle': '0', 'Aquisition order': '0-flip', 'Microscope': 'Leica', 'Magnification': 'x5'}
```

1. Optionnal: select a subset
--
In case you don't want to run the pipeline on all images, you can select a subset
of susbtrates and associated day at anytime.
>>> hierarchy = {
    "Old-1": ["D4"],
    "Old-2": ["D6"]
}
>>> subset_substrates, subset_days = exp.read_hierarchy(hierarchy)
>>> print("substrates=", subset_substrates, "days=", subset_days)


2. Crop & Refine crop
--
In order to perform organoid analysis, we need to keep only one hive per image (to
ensure each organoid will be measured only once). This step aims at running the crop
& refine crop methods for all experiment to get only the current hive as ROI in each
image.
>>> exp.crop(save_montage=True)
>>> exp.optimized_crop()

3. Computing threshold and generating organoid mask
--
Now that the ROI are selected, we want to measure informations only from the organoids.
Therefore, we'll compute the threshold for each organoid and generates the associated
mask and overlay. The mask generation method is quite cumbersome and long to run,
about 30sec per hive. Note that the threshold computation depends on the channel.
>>> exp.threshold(hierarchy=hierarchy)
>>> exp.generate_organoid_mask(hierarchy=hierarchy)

4. Recover organoid data & remove organoids
--
Once the masks are generated, we can recover the data from the organoid. The results will
be aggregated for all organoids of one susbtrate for each day.
>>> exp.get_organoid_data(hierarchy=None)
As we ran the data collection on the enitre experiment, it means we generated 6 result
files: one for Old-1 susbtrate at D1, another for D4 and another for D6, and the same
ones but for Old-2 susbtrate. Each file is named `Results.csv` and is saved in the Day
folder of the associated substrate.
```<put pitcure of a csv file>
```
The file contains many information regarding organoid. Each row corresponds to one organoids
and its information are contained in the different columns. The metrics extracted can be checked
in the documentation of `Hive.get_save_organoid_data()`. Note that all metrics linked to dimension
(expect centroids and bbox) are in µm.

Now that the data are extracted, we can discard the organoids that have been baddly segmented. To do
so, we 1st need to create a montage of the Overlay pictures that display both the original image and 
the organoid mask. Let's run it on the subset.
>>> exp.create_montage(ImageType.OVERLAY, hierarchy=hierarchy)
The overlay are saved as Montage_Overlay_Old-1_D4_BF.jpeg and Montage_Overlay_Old-2_D6_BF.jpeg
respectively in `Old-1/D4` and `Old-2/D6` folders. You may now inspect segmentation to evaluate
its accuracy and discard organoids that you consider not correctly segmented enough
>>> exp.manual_discard(hierarchy=hierarchy)
```<put pitcures of manual discard>
```
By right-clicking on an organoid, you mat discard it: a cyan cross will overlay the discerded organoid.
You also may un-discard it by right-clicking on it again: the cyan cross diseapears. To save the your
discards, you simply need to close the window. A file named `UpdatedResults.csv` will be created. This
file contains the same information and is structured in similat manner than `Results.csv`, but contains
updated values for the `Manually discarded` column: the value is set to `True` for the discarded organoids.

5. Manually modify organoid mask
--
In case you would like to measure an organoid that have been baddly segmented and that you prefered to
discard, you may manually modify the mask. To do so, open you Mask and Optimized_Shadow image on the
software of you choice, modify the mask and save it as an 8-bit grayscale 'tif' with the exact same name
in the exact same location.
Here is an example using Fiji (ImageJ) open-source software:
1. Open the Mask and Optimized_Shadow image of the same hive
2. Select the Mask image and select `Image > Overlay > Add Image`
3. In the `Image to add` field, select the Optimized_Shadow image and set `Opacity` to 70%
4. Using the brush tool with white color, manually draw the mask. Note that you may change
the brush size by righ-clicking on it.
5. Once the mask is correct, select the Mask image (containg now also the Optimized_Shadow
image) and go to `Image > Overlay > Remove Overlay`
6. Now that you only have the mask, you may close the window. It will ask you if you want to
save it : press `Save`. It will ask you if you want to replace it. If you decide not to 
replace it, keep in mind to have your updated mask with the name of the original mask and to
either rename your original mask or store it in a sub-folder.
You may know re-run the following methods on the appropriate substet:
>>> exp.get_organoid_data(hierarchy=hierarchy)
>>> exp.create_montage(ImageType.OVERLAY, hierarchy=hierarchy)
>>> exp.manual_discard(hierarchy=hierarchy)

6. Display organoid evolution over time
--
In order to have some visualization of your experiment, you may select some hives for each
desired substrate and display their pictures at different timepoint. To do so, you first 
need to declare your subset. This subset is structures as a dictionnary of `Exp.name` keys.
Each value is another dictionnary where each key is a `Susbtrate.name` and each value of
this dictionnay is a list of hive numbers. Note that the method will display empty image
in case it do not finds the requested hive.
Here we want to display organoids from both susbtrates at D1, D4 and D6. We decide to follow
hives 0, 9 and 18 for substrate Old-1 and hives 5, 6 and 9 for substrate Old-2. We want to 
show only the BF image from the Optimized_Shadow ones.
>>> hierarchy = {
    "Exp009-B": {
        "Old-1": [0,9,18],
        "Old-2": [5,6,9]
    }
}
>>> exp.display_dynamic(hierarchy, ["D1", "D4", "D6"], (0,0,0,ChannelNames.BF.name), ImageType.OPTIMIZED_SHADOW)
````<put picture created>
```
You may now save this picture directly from the matplolib figure created. As the fontsize
may not be perfrectly adjusted, it is recommanded to save it as vectorial image for later
adjustements on fontsize, colors, ...
Note that you may use a similar method to display organoid evlotuion over time among
mulitple experiments. In this case, you need to declare as many instances of Exp class
that you have experiments and use the method `display_montage_organoids_dynamic`. This
method has the same attributes than `Exp.display_dynamic` plus one: which Exp instances
to use.
exmaple: exp1 and exp2 are Exp instances.
>>> display_montage_organoids_dynamic(hierarchy, [exp1, exp2], ["D1", "D4", "D6"], (0,0,0,ChannelNames.BF.name), ImageType.OPTIMIZED_SHADOW)
"""
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

def unique_list(non_unique_list: List[str]|List[int]):

    """Get unique value of a 1D list and keep the order of appearence of each unique value"""

    unique_list = []
    for elt in non_unique_list:
        if elt not in unique_list:
            unique_list.append(elt)

    return unique_list

def natural_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', s)]

def flatten_list_recursive(variable_list, final='dict'):

    """Flatten a unlimited nested list using recusrive methode"""

    out_list = []

    for i,elt in enumerate(variable_list):
        if isinstance(elt, list):
            elt_out = flatten_list_recursive(elt)
        else:
            if final == "dict":
                elt_out = elt.values()
            elif final == "str":
                elt_out = elt
        
        out_list += elt_out
        
    return out_list

def get_all_attribute_names(obj):
    names = set(vars(obj).keys())  # instance attributes

    # Add property names
    for name, value in obj.__class__.__dict__.items():
        if isinstance(value, property):
            names.add(name)

    return sorted(names)

def image_number_from_filename(filename, format='int', pattern="00[0-1][0-9]"):
    """Returns the number written in the image as int by default or in string if format=='str'."""
    
    number = re.search(pattern, filename).group()
    if format == 'int':
        return int(number)
    elif format == 'str':
        return str(number)
    else:
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
    

    if design in ["H2M100", "H2M150", "H2M200", "H2M150b"]:
        height = 2000
        n_expect = 19
    elif design == 1:
        height = 1000
        n_expect = 19
    elif design == 0: #for test
        height = 2000
        n_expect = 1000
    elif design == "Plate":
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

        case "Zeiss-Incubator":
            match magnification:
                case "x5":
                    if dimension == "2048x2048":
                        um2pixel = 0.675
                        pxl2um = 1.48
    
    return height, n_expect, um2pixel

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
    
def display_montage_organoids_dynamic(hierarchy: dict, obj: List[Exp], days: List[str], slices: tuple[int, int, int, ChannelNames], image_type: ImageType, ext=None):
    """
        hierachy = {
            expI: {
                substrateI: ["hive_number_I", "hive_number_I", "hive_number_3"], #write hive numbers as number ie
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
            current_days = [elt.name for elt in exp.substrates_refs[idx_s_name].days_refs]
            for idx_d_name, d_name in enumerate(current_days):
                hierarchy_obj[exp.name][s_name][d_name] = exp.substrates_refs[idx_s_name].days_refs[idx_d_name].hive_number
    # print("Hierarchy obj", hierarchy_obj)
    
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
    # print(hierarchy, "\n", hierarchy_real)

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
    print(nrows, ncols)
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
    img_tmp = hive_tmp.open_image(image_type, slices, rgb=True)
    dim = img_tmp.shape
    img_empty = np.zeros(dim)
    del hive_tmp, img_tmp, dim
    
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
                        current_days = [elt.name for elt in obj[idx_e].substrates_refs[idx_s].days_refs]
                        idx_d =  current_days.index(d)
                        idx_h = obj[idx_e].substrates_refs[idx_s].days_refs[idx_d].hive_number.index(h)
                        hive = obj[idx_e].substrates_refs[idx_s].days_refs[idx_d].hives_refs[idx_h]
                        img = hive.open_image(image_type, slices, rgb=True)
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
    complete_path = path + path_delimiter + name + '.jpg'
    fig.savefig(complete_path, dpi=300)

def intensity_mad(regionmask, intensity_image):
    return median_abs_deviation(intensity_image[regionmask])

def intensity_median(regionmask, intensity_image):
    return np.median(intensity_image[regionmask])

def get_filenames_leica(filename_pattern, path, exp, substrate, day, image_type):
    """Leica images have only one channel (BF), no z-stack, no timelaps only one image (region) per hive"""
    
    #Initiate output
    filenames = []
    image_number = [] #only for hive level
    
    #Adjust filename pattern:
    params = {"exp": exp, "substrate": substrate, "day": day}
    pattern = adjust_filename_pattern_leica(filename_pattern, image_type, params)

    other_pattern = ImageType.get_attr()
    if image_type == "":
        del other_pattern[0]
    else:
        del other_pattern[0]
        other_pattern.remove(image_type)

    #Search for files
    folder_path = path 
    if os.path.isdir(folder_path) == False:
        warnings.warn(f"{folder_path} do not exist.", UserWarning)
    else:
        files = os.listdir(folder_path)
        files.sort()

        for f in files:
            if (pattern in f) and (any(ext in f for ext in ImageFormat.get_attr()) and any(ty in f for ty in other_pattern)==False) :
                filenames.append([[[{ChannelNames.BF.name: f}]]])
                image_number.append(image_number_from_filename(f, format='int', pattern="[0-9][0-9][0-9][0-9]"))

    return filenames, image_number

def get_filenames_zeiss_incub_pos(path, image_type):

    filenames = []
    image_number = []
    
    #folder_path = path + path_delimiter + substrate + path_delimiter + day
    folder_path = path

    pattern = "img_channel00[0-9]_position[0-9][0-9][0-9]_time[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_z[0-9][0-9][0-9].tif"
    
    if os.path.isdir(folder_path) == False:
        warnings.warn(f"{folder_path} do not exist.", UserWarning)

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
        
        files = os.listdir(folder_path + path_delimiter + folder)
        if len(files) > 0:

            #Determine image dimension (number of timepoints, z-stacks, channel)
            metadata = json.load(open(folder_path + path_delimiter + folder + path_delimiter + "metadata.txt"))
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
                                filenames[-1][s][t][z][c] = folder + path_delimiter + f
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
    
    #folder_path = path + path_delimiter + substrate + path_delimiter + day
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
        return perimeter/np.pi   
    
    def wrinkling_index(equivalent_diameter_perimeter, equivalent_diameter_area):     
        return equivalent_diameter_perimeter/equivalent_diameter_area

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
    def hexagon_outline_ndarray(cls, image_shape, center, side, thickness=1, return_type='hexagon'):
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

        return img

class Exp:

    substrates_refs: list[Substrate] #enable autocompletion and will hold Substrate instances
    
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
        
        print("Creating Exp instance ....", end="\r")

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
            raise ValueError(f"Experiment folder {path} do not exist.")
        
        #Create hexagon instance for each Microscope/Magnification/ImageDim encoutered
        self.hexagon = self.create_hexagon_masks()   
        
        #Create and store daughter class
        for s in substrates:
            substrate = Substrate(s, self, days, params)
            self.substrates_refs.append(substrate)

        print("Creating Exp instance done.", end="\r")
        print()

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
        - Rescaled masks are computed using the global `scale_ratio`.
        - These masks are accessed at Day and Hive levels through
        parent references (self.parent / self.parent.parent).
        - Must be called before cropping or rotation-based alignment.
        """

        hexagon_mask = {}
        
        hexagon_params = unique_list([(elt["Design"],elt["Microscope"],elt["Magnification"],elt["Dimension"]) for elt in self.params])
        
        for elt in hexagon_params:

            img_shape = [ int(i) for i in elt[3].split("x") ]
            center = get_img_center(img_shape)
            hive_height, n_expect, um2pxl = assign_dim(elt[0], elt[1], elt[2], elt[3])
            side = HexagonMath.get_side_from_height(hive_height)*um2pxl
            img_shape_rs = [ round(i/scale_ratio) for i in img_shape]
            center_rs = get_img_center(img_shape_rs)
            side_rs = HexagonMath.get_side_from_height(hive_height/scale_ratio)*um2pxl
            hexagon_mask[elt] = {
                "hexagon_full_scale": HexagonMath.hexagon_outline_ndarray(img_shape, center, side, thickness=5, return_type='hexagon'),
                "hexagon_rescale": HexagonMath.hexagon_outline_ndarray(img_shape_rs, center_rs, side_rs, thickness=5, return_type='hexagon')
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
                substrates.append(s_name)
                days.append(hierarchy[s_name])

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
        # print(substrates, days)
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

    def threshold(self, display='off', verbose='off', hierarchy: dict=None):
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
                s.threshold_hives(display='off', verbose='off', days=days[idx_s]) #Threshold organoid

    def get_organoid_mask(self, channel: ChannelNames=None, hierarchy: dict=None):

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
                s.generate_organoid_mask(channel=channel, display='off', verbose='off', days=days[idx_s]) #Organoid organoid

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

    def get_save_organoid_data(self, properties=OrganoidProperties.rg_properties.value, additionnal_props=OrganoidProperties.additionnal_props.value, metadata=OrganoidProperties.metadata.value, days: List[str]=None,  hierarchy: dict=None, ResultFile="Results.csv"):

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
                s.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, days=days[idx_s], ResultFile=ResultFile) #Organoid organoid
  
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
        """Defines how to print the Substrate class: Substrate(SubstrateName, ExpName, [D1, ..., DN])"""
        days = ",".join(self.days_names)
        rep = "Substrate(" + self.name + ", " + self.parent.name + ", " + days + ")"
        return rep

    def display_hives_dynamic(self, hierarchy: dict, days: list, image_type: ImageType, ext=None):
        """Display the pictures of selected hives and substrates at requested times. The non-found requests will display empty images.
        
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
        display_montage_organoids_dynamic(hierarchy, [self.parent], days, image_type, ext)

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

    def threshold_hives(self, display='off', verbose='off', days: List[str]=None):
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
                    d.threshold_hives(display=display, verbose=verbose)
            else:
                d.threshold_hives(display=display, verbose=verbose)

    def generate_organoid_mask(self, channel: ChannelNames=None, display='off', verbose='off', days: List[str]=None):

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
                    d.generate_organoid_mask(channel=channel, display=display, verbose=verbose)
            else:
                d.generate_organoid_mask(display=display, verbose=verbose)

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

    def get_save_organoid_data(self, properties=OrganoidProperties.rg_properties.value, additionnal_props=OrganoidProperties.additionnal_props.value, metadata=OrganoidProperties.metadata.value, ResultFile="Result.csv", days: List[str]=None):

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
                    d.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, ResultFile=ResultFile)
                    #d.save_organoid_data(data, headers=properties+additionnal_props+metadata, ResultFile=ResultFile)
            else:
                d.get_save_organoid_data(properties=properties, additionnal_props=additionnal_props, metadata=metadata, ResultFile=ResultFile)
                #d.save_organoid_data(data, headers=properties+additionnal_props+metadata, ResultFile=ResultFile)

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
        self.filenames_original, self.image_number = self.get_filenames(ImageType.ORIGINAL.value, current_param["Microscope"])
        self.filenames_crop, _ = self.get_filenames(ImageType.CROP.value, current_param["Microscope"])
        self.filenames_optimized_shadow, _ = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value, current_param["Microscope"])
        self.filenames_mask, _ = self.get_filenames(ImageType.MASK.value, current_param["Microscope"])
        self.filenames_overlay, _ = self.get_filenames(ImageType.OVERLAY.value, current_param["Microscope"])
        self.filenames_core_mask, _ = self.get_filenames(ImageType.CORE_MASK.value, current_param["Microscope"])
        self.filenames_shell_mask, _ = self.get_filenames(ImageType.SHELL_MASK.value, current_param["Microscope"])

        if self.filenames_original is None:
            # raise UserWarning(f"{self.path} do not contain any usable picture.")
            warnings.warn(f"{self.path} do not contain any usable picture.", UserWarning)
            return

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
        hexagon_param = (self.current_param["Design"],self.current_param["Microscope"],self.current_param["Magnification"],self.current_param["Dimension"])
        self.hive_mask = self.parent.parent.hexagon[hexagon_param]["hexagon_full_scale"]
        #Create associated rescaled hive mask by scale_ratio
        self.hive_mask_rs = self.parent.parent.hexagon[hexagon_param]["hexagon_rescale"]

        #Generate Hive Class
        idx = idx_acquisition_order(self.current_param["Aquisition order"])
        idx_flat = [item for sublist in idx for item in sublist]
        for i,file in enumerate(self.filenames_original):
            self.hive_number.append(get_hive_number(self.image_number[i], self.current_param["Aquisition order"]))
            hive = Hive(self.hive_number[-1], self.image_number[i], self, file)
            self.hives_refs.append(hive)
           
    def __repr__(self):
        """Defines how to print the Day class: Day(DayName, SubstrateNmae, ExpName, Aquisition Parameters)"""
        rep = "Day(" + self.name + ", " + self.parent.name + ", " + self.parent.parent.name + ", " +  repr(self.current_param) + ")"
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
        folder_path = self.parent.parent.path + path_delimiter + self.parent.name + path_delimiter + self.name
        filenames = self.get_attribute_name(image_type, 'filenames_', '')
        filenames_flat = flatten_list_recursive(filenames, final='dict') 
        for files in filenames_flat:
            with Image.open(folder_path + path_delimiter + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            raise RuntimeError(f"{image_type.name}  images don't have same dimension in {folder_path}")
        elif len(set(dims)) == 1:
            dim_name =f"img_{image_type.name.lower()}_dim"
            setattr(self, dim_name, (dims[0][1], dims[0][0]) )
        else:
            raise RuntimeError(f"{folder_path} issue, no {image_type.name} images found")
    
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

        filename_pattern = self.parent.parent.filename_pattern
        path = self.path
        exp_name = self.parent.parent.name
        substrate_name = self.parent.name
        day = self.name

        if microscope == "Leica":
            filenames, image_number = get_filenames_leica(filename_pattern, path, exp_name, substrate_name, day, image_type)
        
        elif microscope == "Zeiss-Incubator":
            filenames, image_number = get_filenames_zeiss_incub_pos(path, image_type)

        elif microscope == "Zeiss-confocal":
            filenames, image_number = get_filenames_zeiss_confocal_czi(image_type)

        else:
            warnings.warn("Not implemented yet.")

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
            updated_pattern = adjust_filename_pattern_leica(pattern, image_type, params)
        
        elif microscope == "Zeiss-Incubator":
            print("Zeiss-Incubator")
            updated_pattern = adjust_filename_pattern_zeiss_incub_pos(pattern, image_type, params)

        return updated_pattern
    
    def find_rotation_angle(self, scale_ratio, slices):
        """Return the rotation angle of the hives for the current day and substrate.
        
            For each hive of the current substrate and day, determine the rotation
            angle within the range ±5° from ``self.current_param["Angle"]`` (see
            ``Hive.find_hive_rotation_angle`` for more information). The rotation
            angle of the imaging session is then defined as the median of the
            rotation angles determined for each hive.

            Parameters
            ----------
            scale_ratio : int
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
        best_angles = []
        n_hives = len(self.hives_refs)
        for i, hive in enumerate(self.hives_refs):
            print(f"{self.parent.name}_{self.name}, Computing best angle... {i}/{n_hives}", end="\r")
            best_angles.append(hive.find_hive_rotation_angle(self.hive_mask_rs, int(self.current_param["Angle"]), 5, 0.5, scale_ratio, slices)) #Compute & store best angle
        best_angle = np.nanmedian(best_angles) #select best angle from set
        print(f"{self.parent.name}_{self.name}, Computing best angle done: {best_angle}°", end="\r")
        print()
        return best_angle

    def create_montage(self, image_type: ImageType, channel: ChannelNames=None, slices: Tuple[int,int,int,ChannelNames]=None, display_hive_number='on', scale_ratio: int=6):
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

        scale_ratio : int, default=6
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

        if slices is None:
            slices = self.get_slices_range()
        else:
            slices = ([slices[0]], [slices[1]], [slices[2]], [slices[3]])
        
        if channel is None:
            channel = slices[3]
        else:
            channel = [channel]

        #Image acquisition order:
        idx = idx_acquisition_order(self.current_param["Aquisition order"])

        for s in slices[0]:
            for t in slices[1]:
                for z in slices[2]:
                    for c_i, c in enumerate(channel):

                        #Adjust filename pattern:
                        params = {
                            "exp": self.parent.parent.name, "substrate": self.parent.name, "day": self.name, 
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
                        
                        pattern = self.adjust_filename_pattern(self.current_param["Microscope"], self.parent.parent.filename_pattern, image_type.value, params)

                        #If missing pictures, adding empty ones
                        img, filenames = add_empty_img(img, filenames_flat, image_type.value + pattern, 19)

                        #Get coordinate for montage
                        img_sz = img[0].shape
                        img_type = img[0].dtype
                        coord, idx_flat = get_hive_coord_montage(img_sz, idx, base='hive')

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
                                    draw.text((coord[i][0][0]+25, coord[i][1][0]+50), str(get_hive_number(img_number, self.current_param["Aquisition order"])), font=font, fill=(255,255,255))
                                else:
                                    draw.text((coord[i][0][0]+25, coord[i][1][0]+50), str(get_hive_number(img_number, self.current_param["Aquisition order"])), font=font)
                            #montage_pil.show()
                            montage = np.array(montage_pil)
                            montage_type = montage.dtype

                        #Downsize image
                        if len(montage.shape) > 2:
                            montage = downscale_local_mean(montage, (scale_ratio, scale_ratio,1))
                        else:
                            montage = downscale_local_mean(montage, (scale_ratio, scale_ratio))
                        montage = montage.astype(montage_type)

                        #Transform in 8bit
                        montage = img_as_ubyte(montage)

                        #Save montage
                        if image_type is ImageType.CROP:
                            out_path = self.parent.parent.path + path_delimiter + \
                            "Montage_" + image_type.value + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + "_" + c + ".jpg"
                        else:
                            out_path = self.path + path_delimiter + \
                            "Montage_" + image_type.value + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + "_" + c + ".jpg"
                        imsave(out_path, montage)

        print(f"Montage image saved: {out_path}")

        return montage

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
        scale_ratio : int
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
                    best_angle = self.find_rotation_angle(scale_ratio, (s,t,z,ChannelNames.BF.name))

                    for c in slices[3]:

                        #Crop & rotate image
                        for i,hive in enumerate(self.hives_refs):
                            print(f"{self.parent.name}_{self.name}, Cropping pictures... {i}/{n_hives}", end="\r")
                            hive.crop_hive(scale_ratio, best_angle, slices=(s,t,z,c), save=True)

        #Update Crop filenames
        self.filenames_crop, _ = self.get_filenames(ImageType.CROP.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_crop = hive.update_filename(ImageType.CROP)

        #Verify crop pictures have same dimension
        folder_path = self.parent.parent.path + path_delimiter + self.parent.name + path_delimiter + self.name
        dims = []
        for files in flatten_list_recursive(self.filenames_crop):
            with Image.open(folder_path + path_delimiter + files) as img:
                dims.append(img.size)
        if len(set(dims)) > 1:
            raise RuntimeError(f"Cropped images don't have same dimension in {folder_path}")
        self.img_crop_dim = (dims[0][1], dims[0][0])

        print(f"{self.parent.name}_{self.name}, Croping pictures done: {self.img_crop_dim}", end="\r")
        print()

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
            warnings.warn(f"No Crop images found for {self.parent.name}-{self.name}")
            return
        
        #Generate the concentric hexagon sides
        for j,k in enumerate(range(k_step, n_slice*k_step+k_step, k_step)):
            side_k = round((height_pxl - 2*k)/(2*np.sin(np.pi/3)))
            hives_mask_side[j] = hexagon_outline_ndarray(self.img_crop_dim, [round(self.img_crop_dim[0]/2), round(self.img_crop_dim[1]/2)], side_k, thickness=k_step, return_type='side')
        
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
                            print(f"{self.parent.name}_{self.name}, Refine cropping... {j}/{n_hives}", end="\r")
                            hive.optimized_crop_hive(hives_mask_side, k_step, (s,t,z,c), display=display)

        #Update filenames
        self.filenames_optimized_shadow, _ = self.get_filenames(ImageType.OPTIMIZED_SHADOW.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_optimizedshadow = hive.update_filename(ImageType.OPTIMIZED_SHADOW)

        print(f"{self.parent.name}_{self.name}, Refine cropping done.       ", end="\r")
        print()

    def threshold_hives(self, slices: Tuple[int,int,int,ChannelNames]=None, display='off', verbose='off'):

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

        #Verify the optimized_shadow image exists
        if len(self.filenames_optimized_shadow) == 0:
            warnings.warn(f"No Optimized_Shadow images found for {self.parent.name}-{self.name}")
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
                            n_r, n_c = get_compact_grid(n_hives) #adequate subplot grid
                            fig, ax = plt.subplots(nrows=n_r, ncols=n_c, sharex=True, sharey=True, 
                                                num=f"Threshold for {self.parent.parent.name}_{self.parent.name}_{self.name}_{c}") #create subplots
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
                            print(f"{self.parent.name}_{self.name}_{c}, Determining threshold... {j}/{n_hives}", end="\r")
                            if display == 'on':
                                current_ax =  ax[idx_ax[j]]
                            else:
                                current_ax = None
                            if c == ChannelNames.BF.name:
                                thresholds[j][s][t][z][c] = hive.custom_threshold_hive((s,t,z,c), fig1=fig, ax1=current_ax, title=f"#{hive.hive_number}", verbose=verbose)
                            else:
                                thresholds[j][s][t][z][c] = hive.otsu((s,t,z,c))

                        print(f"{self.parent.name}_{self.name}_{c}, Determining threshold done.    ", end="\r")
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

    def generate_organoid_mask(self, channel: ChannelNames=None, slices: Tuple[int,int,int,ChannelNames]=None, display='off', verbose='off', manual_discard='off'):
        
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
        if len(self.filenames_optimized_shadow) == 0:
            warnings.warn(f"{self.parent.name}_{self.name}, No optimized images found. Cannot generate masks.")
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
                            fig, ax = plt.subplots(ncols=n_hives, nrows=4, num=f"Segmentation Steps - {self.parent.parent.name}_{self.parent.name}_{self.name}_{c}")
                            ax[0, 0].set_ylabel("Original")
                            ax[1, 0].set_ylabel("Threshold")
                            ax[2, 0].set_ylabel("Binary operations")
                            ax[3, 0].set_ylabel("Mask")
                        else:
                            fig=None
                            ax=None

                        #Generate mask & overlay
                        for j,hive in enumerate(self.hives_refs):
                            print(f"{self.parent.name}_{self.name}_{c}, Generating organoid mask... {j}/{n_hives}", end="\r")
                            if display == 'on':
                                current_ax_idx =  j
                            else:
                                current_ax_idx = None
                            hive.generate_organoid_mask((s,t,z,c), fig=fig, ax=ax, ax_idx=current_ax_idx, verbose=verbose)

        print(f"{self.parent.name}_{self.name}_{c}, Generating organoid mask done.    ", end="\r")
        print()

        #Update filenames
        self.filenames_mask = self.get_filenames(ImageType.MASK.value, self.current_param["Microscope"])
        self.filenames_overlay = self.get_filenames(ImageType.OVERLAY.value, self.current_param["Microscope"])
        for hive in self.hives_refs:
            hive.filenames_mask = hive.update_filename(ImageType.MASK)
            hive.filenames_overlay = hive.update_filename(ImageType.OVERLAY)

        #Display
        if display == 'on':
            plt.show()

        if manual_discard == 'on':
            self.manual_discard()

    def get_save_organoid_data(self, slices: Tuple[int,int,int,ChannelNames]=None, properties=OrganoidProperties.rg_properties.value, additionnal_props=OrganoidProperties.additionnal_props.value, metadata=OrganoidProperties.metadata.value, ResultFile="Results.csv"):

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
                        print(f"{self.parent.name}_{self.name}, Getting organoid data... {i}/{n_hives}", end="\r")
                        data_hive, organoid_count = hive.get_organoid_data(organoid_count, (s,t,z,slices[3]), properties=properties, additionnal_props=additionnal_props, metadata=metadata)
                        data.append(data_hive)
        print(f"{self.parent.name}_{self.name}, Getting organoid data done.    ", end="\r")
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
        print(f"{self.parent.name}_{self.name}, Organoid data saved.    ")

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

        full_path = self.path + path_delimiter + ResultFile
        
        #Open csv file to store segmentation results
        f = open(full_path, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        #Save data
        for row in data:
            writer.writerows(row)

        #Close file
        f.close()

    def manual_discard(self, overlay=None, newResultFile="UpdatedResults.csv", ResultFile="Results.csv", scale_ratio: int=6):

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

        scale_ratio : int, default=6
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

        #Verify Overlay montage exist
        overlay_path = self.path + path_delimiter +  "Montage_" + ImageType.OVERLAY.value + self.parent.parent.name + "_" + self.parent.name + "_" + self.name + "_" + ChannelNames.BF.name + ".jpg"
        if os.path.isfile(overlay_path) == False:
            warnings.warn(f"Montage Overlay not found for {self.parent.name}-{self.name}")
            return
        
        #Open overlay image
        if overlay is None:
            overlay = np.array(imread(overlay_path))
        else:
            overlay = overlay

        #Read and store data
        ResultFile_path = self.path + path_delimiter + ResultFile
        if os.path.isfile(ResultFile_path) == False:
            return ValueError(f"Result file {ResultFile_path} do not exist.")
        f = open(self.path + path_delimiter + ResultFile, 'r', newline='')
        spamreader = csv.reader(f, delimiter=',')
        data=[]
        for i,row in enumerate(spamreader):
            if i == 0: headers = row
            else:
                data.append({})
                [data[-1].update({headers[j]: row[j]}) for j in range(len(headers))]
        f.close()

        #Get index for aquisition order
        idx = idx_acquisition_order(self.current_param["Aquisition order"])
        #Get hive coordinates
        coord_hives, idx_flatten = get_hive_coord_montage(overlay.shape, idx, base='montage')
        #Rescale coordinates due to rescale
        coords = []
        for i in range(len(data)):
            bbox = (
                float(data[i]["bbox_ymin"])/scale_ratio, 
                float(data[i]["bbox_xmin"])/scale_ratio, 
                float(data[i]["bbox_ymax"])/scale_ratio,  
                float(data[i]["bbox_xmax"])/scale_ratio
                )
            coords.append({"Index": data[i]["Index"], \
                            "Hive number": data[i]["Image number"], \
                            "Hive coord": coord_hives[idx_flatten.index(int(data[i]["Image number"]))],
                            "bbox": bbox 
                            })
        
        #Create figure
        fig_name = f"{self.parent.name}_{self.name} Manual Discard"
        fig3, ax3 = plt.subplots(num=fig_name)
        #Remove ticks label
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.imshow(overlay)
        discarded_organoids = [] #will hold the discarded hives
        handler_on_click = partial(on_click_discard_undiscard_organoid, coords=coords, discarded_organoids=discarded_organoids) #to enable passing variable to plt.connect functions
        fig3.canvas.mpl_connect('button_press_event', handler_on_click) #callbaks to discard/undiscard hives when selecting a hive on the picture
        handler_on_close = partial(save_before_close, path=self.path, name=fig_name)
        fig3.canvas.mpl_connect('close_event', handler_on_close)
        plt.show()

        #Save Manual Discard image

        #Once picture is closed, update data & update results file
        f = open(self.path + path_delimiter + newResultFile, 'w+', newline='')
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
        print(f"{self.parent.name}_{self.name}: Discarded organoids are: {discarded_organoids}")   

class Hive:

    parent: Day #enable autocompletion and will hold the calling Day instance

    def __init__(self, hive_number, image_number, parent, filenames_original):

        self.hive_number = hive_number
        self.image_number = image_number
        self.parent = parent
        self.filenames_original = filenames_original
        self.dimension = {
            "site": len(filenames_original), 
            "timepoint": len(filenames_original[0]), 
            "z-stack": len(filenames_original[0][0]),
            "n_channel": len(filenames_original[0][0][0]),
            "channel": filenames_original[0][0][0]}
        self.name = self.parent.parent.parent.name + "_" + self.parent.parent.name + "_" + self.parent.name + "_" + str(self.hive_number)
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

    def update_filename(self, image_type: ImageType):
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
            index_hive_in_Day = self.parent.filenames_original.index(self.filenames_original)
            filename_attribute_name = self.parent.get_attribute_name(image_type, "filenames_", "")
            return filename_attribute_name[index_hive_in_Day]
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

        #Determine filename & check it exists
        file_name = self.get_attribute_name(image_type=image_type, prefixe="filenames_", suffixe="")
        filepath = self.parent.parent.parent.path + path_delimiter + self.parent.parent.name + path_delimiter + \
            self.parent.name + path_delimiter + file_name[slices[0]][slices[1]][slices[2]][slices[3]]
        if os.path.isfile(filepath) == False:
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
    
    def find_hive_rotation_angle(self, hive_mask, angle_user, angle_range, angle_step, scale_ratio, slices):

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

        scale_ratio : int
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
        - The hive image is resized by `scale_ratio` to accelerate computation.
        - Convolution is computed using FFT convolution (`fftconvolve`).
        - The method assumes `self.open_image` correctly loads and converts the
        image to grayscale if needed.
        """
        #Open image
        img = self.open_image(ImageType.ORIGINAL, slices)
        img = resize(img, (round(img.shape[0]/scale_ratio), round(img.shape[1]/scale_ratio)))
        
        angles = np.arange(angle_user-angle_range, angle_user+angle_range, angle_step)
        conv_max_value = [] #will hold list on maximum value of each convolution output

        for a in angles:
            conv = fftconvolve(img, rotate(hive_mask, a), mode='same') #compute convolution of hive image with a rotation of hive mask
            conv_max_value.append(conv.flatten().max()) 

        return angles[conv_max_value.index(max(conv_max_value))]
       
    def crop_hive(self, scale_ratio, best_angle, slices: Tuple[int, int, int, ChannelNames], save=True):
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
        scale_ratio : int
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

        #Open image
        img = self.open_image(ImageType.ORIGINAL, slices, rgb=True) #open image
        img_type = img.dtype

        #rotate image
        img = rotate(img, -best_angle, preserve_range=True, resize=False) #weird, before needed to write resize=True and now need o write False
        img = img.astype(img_type)
        img = adjust_dimension_after_rotation(img, [round(self.parent.hive_height*self.parent.um2pxl), round(self.parent.hive_diag*self.parent.um2pxl)], img_type) #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)

        if ChannelNames.BF.name == slices[3]:

            (y_c, x_c) = self.get_hive_barycenter(best_angle, self.parent.hive_mask_rs, scale_ratio, ImageType.ORIGINAL, slices) #compute hive barycenter

            #Compute x and y values where crop should occure:
            y_cut = self.cut_value_for_crop(y_c, self.parent.hive_height*self.parent.um2pxl/2, img.shape[0])
            x_cut = self.cut_value_for_crop(x_c, round(self.parent.hive_diag*self.parent.um2pxl/2), img.shape[1]) #round here to get consistent dimension over pictures

            # Fill the hexagon mask with 1 value
            hive = HexagonMath.hexagon_outline_ndarray(img.shape, (y_c, x_c), self.parent.hive_side*self.parent.um2pxl, thickness=5, return_type='hexagon')
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
            if path_delimiter in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
                filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(path_delimiter)
                sub_folder = filename[0] + path_delimiter
                filename = filename[1]
            else:
                sub_folder = ""
                filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
            out_path = self.parent.path + path_delimiter + sub_folder  + ImageType.CROP.value + filename
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
            self.opti_crop_mask = hive_mask

        #Save the Optimized cropped picture
        img[self.opti_crop_mask == 0] = 0
        if path_delimiter in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(path_delimiter)
            sub_folder = filename[0] + path_delimiter
            filename = filename[1]
        else:
            sub_folder = ""
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
        out_path = self.parent.path + path_delimiter + sub_folder + ImageType.OPTIMIZED_SHADOW.value + filename
        if '.jpg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        elif '.jpeg' in out_path:
            out_path = out_path.replace('.jpg', '.tif')
        imsave(out_path, img)

        plt.show()

    def otsu(self, slices: Tuple[int, int, int, ChannelNames]):

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
        img = self.open_image(ImageType.OPTIMIZED_SHADOW, slices, rgb=True)
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

    def custom_threshold_hive(self, slices: Tuple[int, int, int, ChannelNames], fig1=None, ax1=None, title=None, verbose='off'):

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

        if fig1 is None and ax1 is None:
            fig, ax = plt.subplots()
        else:
            fig = fig1
            ax = ax1
        
        #Open image in grayscale
        img = self.open_image(ImageType.OPTIMIZED_SHADOW, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()
        
        #Turn image into range [0,1]
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

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
                print()
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
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
                print()
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
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
                print()
                print("Fit of Low peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
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
                print()
                print("Fit of High peak encoutered RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800 --> fit paramaters set to 0")
                print()
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
        self.threshold[slices[0]][slices[1]][slices[2]][slices[3]] = x_cut

        return x_cut#, low_cut_type, high_cut_type
    
    def generate_organoid_mask(self, slices: Tuple[int, int, int, ChannelNames], fig=None, ax=None, ax_idx=None, verbose='off'):

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

        mesh_pxl = round(60*self.parent.um2pxl/2)     
        #Open image in grayscale
        img = self.open_image(ImageType.OPTIMIZED_SHADOW, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()

        #Turn image into range [0,1]
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

        #Remove artefacts
        r_blur=5*self.parent.um2pxl #to blur unique cells entierly (considered cells about 10µm diameter)
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
            if (rg.eccentricity > 0.88) or (rg.area_filled <= 10000*self.parent.um2pxl) or (rg.area_filled >= 1500000*self.parent.um2pxl):
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
        if path_delimiter in self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]:
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]].split(path_delimiter)
            sub_folder = filename[0] + path_delimiter
            filename = filename[1]
        else:
            sub_folder = ""
            filename = self.filenames_original[slices[0]][slices[1]][slices[2]][slices[3]]
        out_path = self.parent.path + path_delimiter + sub_folder + ImageType.MASK.value + filename
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
        out_path = self.parent.path + path_delimiter +  sub_folder + ImageType.OVERLAY.value + filename
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

    def generate_core_shell_mask(self):
        pass

        # #Open images
        # img_ipsc = H2M150Flip_D1_1.open_image(rgb=False, filename=H2M150Flip_D1_1.filenames_optimizedshadow)
        # mask_ipsc = H2M150Flip_D1_1.open_image(rgb=True, filename=H2M150Flip_D1_1.filenames_mask)
        # dims = img_ipsc.shape
        # print(dims)

        # #Organoid image only
        # organoid_ipsc = np.zeros(dims)
        # for r in range(dims[0]):
        #     for c in range(dims[1]):
        #         if mask_ipsc[r, c] == 0:
        #             organoid_ipsc[r,c] = 0
        #         else:
        #             organoid_ipsc[r,c] = img_ipsc[r,c]
        # # plt.imshow(organoid_ipsc)
        # # plt.show()

        # #Blur
        # organoid_ipsc_blur = gaussian_filter(organoid_ipsc, 5)

        # #Thresh
        # thresh = threshold_otsu(organoid_ipsc_blur[mask_ipsc == 255])
        # organoid_ipsc_dark = organoid_ipsc < thresh

        # #Binary operation
        # organoid_ipsc_dark = binary_erosion(organoid_ipsc_dark, footprint=ellipse(5,5))
        # organoid_ipsc_dark = area_closing(organoid_ipsc_dark)
        # organoid_ipsc_dark = binary_dilation(organoid_ipsc_dark, footprint=ellipse(5,5))

        # # plt.imshow(organoid_ipsc_dark)
        # # plt.show()

        # #Label & regions
        # labels_dark = label(organoid_ipsc_dark)
        # regions_dark = regionprops(labels_dark, organoid_ipsc)
        # [print(rg.label, rg.area_filled) for rg in regions_dark]

        # #Regions to remove
        # rg_to_remove = []
        # for rg in regions_dark:
        #     if rg.area_filled >= dims[0]*dims[1]*0.8: #background
        #         rg_to_remove.append(rg.label)
        #     elif rg.area_filled < 1000: #artfefacts:
        #         rg_to_remove.append(rg.label)
        # print(rg_to_remove)
        # #Remove them
        # for k in sorted(rg_to_remove, reverse=True):
        #     print(k)
        #     del(regions_dark[k-1])
        #     labels_dark[(labels_dark == k)] = 0

        # #Define dark and clear regions
        # max_area = max([rg.area_filled for rg in regions_dark])
        # min_area = min([rg.area_filled for rg in regions_dark])
        # print(max_area, min_area)

        # #Fill the holes in accurate regions
        # for rg in regions_dark:
        #     r_min, c_min, r_max, c_max = rg.bbox
        #     labels[r_min:r_max, c_min:c_max] = rg.image_filled
        
    def get_organoid_data(self, organoid_count, slices: Tuple[int, int, int, ChannelNames], \
                          properties=OrganoidProperties.rg_properties, \
                            additionnal_props=OrganoidProperties.additionnal_props, \
                                metadata=OrganoidProperties.metadata.value):
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
        BF_slice = (slices[0], slices[1], slices[2], ChannelNames.BF.name)
        filename = self.filenames_original[slices[0]][slices[1]][slices[2]][ChannelNames.BF.name]

        #Open BF mask
        mask_img = self.open_image(ImageType.MASK, BF_slice, rgb=True)
        
        #Open original BF images
        img_gray = self.open_image(ImageType.OPTIMIZED_SHADOW, BF_slice, rgb=True)

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
        pxl2um = 1/self.parent.um2pxl
        regions_gray = regionprops(labels, img_gray, extra_properties=(intensity_median,intensity_mad), spacing=(pxl2um, pxl2um))
        regions_fluo = []
        if len(slices[3]) > 1:
            for c_i, c in enumerate(fluo_chan):
                regions_fluo.append(regionprops(labels, img_fluo[c_i], extra_properties=(intensity_median,intensity_mad), spacing=(pxl2um, pxl2um)))
        data = []
        for rg_i, rg in enumerate(regions_gray):
            rg_props = get_all_attribute_names(rg) + list(rg._extra_properties.keys())
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
            data[-1].update({"Index": organoid_count, "Day": self.parent.name, "Substrate": self.parent.parent.name,
                             "Site": slices[0], "Timepoint": slices[1], "Z-Stack": slices[2],
                             "Image Name": filename, "Hive number": self.hive_number, "Image number": self.image_number,
                             "Manually Discarded": False, "Analysis Date": datetime.now(), "Analysis Version": VERSION}) #metadata
            organoid_count += 1 #incremente organoid count

        idx = idx_acquisition_order("bottom-flip")
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
    
    def get_hive_barycenter(self, best_angle, hive_mask, scale_ratio, image_type: ImageType, slices):

        """Compute the barycenter of a hive within an image using convolution.

        This method opens the image, resizes it according to `scale_ratio`, 
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

        scale_ratio : int
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
        img = resize(img, (round(img.shape[0]/scale_ratio), round(img.shape[1]/scale_ratio)))

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

    def get_microvessel_mask_per_z(self, slices: Tuple[int, int, int, ChannelNames]):

        #Open image
        img = self.open_image(ImageType.OPTIMIZED_SHADOW, slices, rgb=True)
        if len(img.shape) > 2:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.copy()
        
        #Convert image to float 64
        if img_gray.dtype != "float64":
            img_gray = img_as_float64(img_gray)

        #Blur image
        radius = self.parent.um2pxl * 5
        img_blur = gaussian_filter(img_gray, radius)

        #Otsu Threshold
        otsu_threshold = threshold_otsu(img_blur)

        #Apply threshold
        img_thresh = img_blur > otsu_threshold

        plt.imshow(img_thresh, cmap='gray')
      