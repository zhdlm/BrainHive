import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import skimage as ski
from scipy.signal import find_peaks, fftconvolve
from scipy.optimize import curve_fit
import os
from PIL import Image
import logging
from math import floor, ceil , sqrt
import re
import multiprocessing as mp
from functools import partial
import pandas as pd

#############################################################################################
# Function definition
#############################################################################################

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
        rr, cc = ski.draw.line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
        # For each pixel in the line, draw a small disk of radius thickness//2
        if return_type == 'coord-side':
            img.append([x0,y0])
            img.append([x1, y1])
        else:
            for r, c in zip(rr, cc):
                if return_type == 'hexagon':
                    dr, dc = ski.draw.disk((r, c), radius=thickness // 2, shape=img.shape)
                    img[dr, dc] = 1
                elif return_type == 'side':
                    dr, dc = ski.draw.disk((r, c), radius=thickness // 2, shape=img[0].shape)
                    img[i][dr, dc] = 1

    del(image_shape, center, side)
    return img

def fine_tuning_crop(img_gray: np.array, hives_mask_side, height: float, k_step: float, n_slice: int, display='off'):

    """Returns a fine tuned mask of the hexagon (np.array) to get rid of borders and shadows"""

    print('Fine tunning of hexagon mask . . .', end='\r')
    #Variables
    img_shape = img_gray.shape #shape of the picture
    x_c = round(img_shape[1]/2) #center of the picture along x axis
    y_c = round(img_shape[0]/2) #center of the picture along y axis
    coord = [] #will hold the coordinates of the vertices of the fine tunned hexagon
    hive_mask = np.zeros(img_gray.shape) #will hold the mask of the fine tuned hexagon
    cut_idx = [None]*6 #will hold the for each of the 6 side of the hexagon the slice where to cut the hexagon

    #To plot the stats
    if display == 'on':
        fig, ax = plt.subplots(nrows=2, ncols=6, sharey='row')

    #Compute the mean and (quartile 3 - quartile 1) length of the pixel values in all slices of each heagon side
    for s_k in range(6): #loop over the different side of the hexagon
        stats = []
        for k in range(len(hives_mask_side)): #loop over each slice of hexagon
            roi = img_gray[hives_mask_side[k][s_k] > 0].flatten()
            #print(roi)
            #Compute statistical info about the pixel of the side s_k and slice kth of 10pixel thick
            stats.append(dict(med=np.median(roi), 
                                q1=np.percentile(roi, 25), 
                                q3=np.percentile(roi, 75), 
                                mean=np.mean(roi),
                                whislo=np.percentile(roi, 5),
                                whishi=np.percentile(roi, 95), 
                                fliers=[], label=str(k*10 + 10)))
            #Extract means of each slice and compute the derivative over the slices
            means = [elt["mean"] for elt in stats]
            deltaQ = [(elt["q3"] - elt["q1"]) for elt in stats]

        #Compute the means and deltaQ variations
        means_derivative = []
        deltaQ_derivative = []
        for k in range(len(means)-1):
            means_derivative.append((float(means[k+1]-means[k])/2))
            # deltaQ_derivative.append((float(deltaQ[k+1]-deltaQ[k]) /2))
            
        #Display the stats as boxplot and derviatives as lineplot
        if display == 'on':
            ax[0, s_k].bxp(stats)
            ax[1, s_k].plot((range(0, n_slice-1, 1)), means_derivative, color='r', label='mean')
            # ax[1, s_k].plot(0.5 + range(0, n_slice-1, 1), deltaQ_derivative, color='b', label='deltaQ')
        #print(means_derivative, deltaQ_derivative)

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
        if display == 'on':
            ax[1, s_k].annotate(f"{cut_idx[s_k]}", xy=(cut_idx[s_k]+0.02, -10),color='r', fontsize=8)

        #Store the coordinates of the fine tuned vertices of the side s_k
        out = hexagon_outline_ndarray(img_shape, [y_c, x_c], round((height - 2*cut_idx[s_k]*k_step)/(2*np.sin(np.pi/3))), thickness=5, return_type='coord-side')
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
        rr, cc = ski.draw.line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
        # For each pixel in the line, draw a small disk of radius thickness//2
        for r, c in zip(rr, cc):
            dr, dc = ski.draw.disk((r, c), radius=5 // 2, shape=hive_mask.shape)
            hive_mask[dr, dc] = 1
    
    print('Fine tunning of hexagon mask done.')
    print(f"\tCut values (x{k_step} pixel): {cut_idx}")
    print(f"\tUpdated vertices coordinates (x,y): {coord}")
        
    return hive_mask

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

def create_montage(user_inputs, img, img_type, n_expected, filenames):
    # Substrate is definied as follow:
    # 1st column: 3 hives
    # 2nd column: 4 hives
    # 3rd column: 5 hives
    # 4th column: 3 hives
    # 5th column: 4 hives

    #Image acquisition order:
    idx = idx_acquisition_order(user_inputs["Acquisition order"])

    #If missing pictures, adding empty ones
    img, filenames = add_empty_img(img, filenames, user_inputs["Filenames"], n_expected)
    #img_rs, filenames = add_empty_img(img_rs, filenames, user_inputs["Filenames"], n_expected)

    #Get coordinate for montage
    img_sz = img[0].shape
    coord, idx_flat = get_hive_coord_montage(img_sz, idx, base='hive')

    #Reconstruct picture
    montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=img_type)
    for i in range(len(img)):
        montage[coord[i][1][0]:coord[i][1][1], coord[i][0][0]:coord[i][0][1], :] = img[idx_flat[i]]

    return montage

def read_usr_input(file):
    """
    read usr inputs from txt file and return it as dictionnary.
    
    Parameters: 
        file: name of the file. The file should be stored in the same folder than this code. The file should be written as:
            key1= value1
            key2= value2
            ...
            ..
            .
            keyN= valueN

    Returns:
        dictionnary: written as:
            {key1: value1, key2: value2, ..., keyN: valueN}
    """

    try:
        user_inputs = open(file).read()
        txt = user_inputs.split('\n')
        user_inputs = {}
        for i in range(len(txt)):
            line = txt[i].split('= ')
            key =  line[0]
            value = line[1]
            sub_values = value.split(', ')
            if value.lstrip('-').isdigit():
                if '-' in value:
                    value = - int(value.lstrip('-'))
                else:
                    value = int(value)
            elif len(sub_values) > 1:
                value = sub_values
            user_inputs.update({key: value})
        print("\nUser inputs loaded")

    except FileNotFoundError:
        user_inputs = {}
        print("\nNot finding file containing user inputs.\nScript Aborted")
        raise
        quit()
    
    except:
        user_inputs = {}
        print("\nError in user inputs, script aborted. File should be stored in the " \
        "folder containing this script and be written as\nkey1= value1\n...\nkeyN= valueN" \
        "\nScript Aborted")
        quit()

    del(file)
    return user_inputs

def assign_param(user_inputs):
    
    match user_inputs["Hive"]:
        case 2:
            height = 2000
            n_expect = 19
        case 1:
            height = 1000
            n_expect = 19
        case 0: #for test
            height = 2000
            n_expect = 1000

    match user_inputs["Microscope"]:

        case "Leica":
            match user_inputs["Magnification"]:
                case "5x":
                    if user_inputs["Dimension"] == "1944x2592":
                        um2pixel = 0.9 # 1 µm corresponds to 0.5645 pxl
                        pxl2um = 1.1 # 1 pxl corresponds to 1.7714 µm
                    elif user_inputs["Dimension"] == "1200x1600":
                        um2pixel = 0.54 # 1 µm corresponds to 0.5645 pxl
                        pxl2um = 1.85 # 1 pxl corresponds to 1.7714 µm
                # case "10x":
                #     if user_inputs["Dimension"] == [1944, 2592]:
                #         pass
                #     elif user_inputs["Dimension"] == [1200, 1600]:
                #         pass
    
    del(user_inputs)
    return height, n_expect, um2pixel

def idx_acquisition_order(acquisition_order):

    # Substrate is definied as follow:
    # 1st column: 3 hives
    # 2nd column: 4 hives
    # 3rd column: 5 hives
    # 4th column: 3 hives
    # 5th column: 4 hives
    
    idx = None

    match acquisition_order:
        case 'bottom':
            idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]]
        case 'bottom-flip':
            idx = [[16, 15, 7], [17, 14, 8, 6], [18, 13, 9, 5, 0], [12, 10, 4, 1], [11, 3, 2]]
        case 'bottom right':
            idx = [[2,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
        case 'exp001':
            idx = [[7,6,5], [15,8,4,0], [16,14,9,3,1], [17,13,10,2], [18,12,11]]
        case 'bottom center':
            pass
            # idx = [[,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
    return idx

def store_img_filename(user_inputs, path_delimiter, n_expect):
    """
    returns name of files from path that are pictures and that contains filename_pattern in their name
    
    Parameters: 
        path (string): directory leading to the folder containing the pictures of interest
        filename_pattern (string): pattern to search search in the name of the files
        n_expect (int): number of pictures expected

    Returns:
        list: each element is a string containing the complete name of the picture
    """
    #Reconstruct the paths and pattern of the filenames
    if 'Days' in user_inputs.keys() and 'Substrate' in user_inputs.keys():
        print(f"days={user_inputs["Days"]}, susbstrate={user_inputs["Substrate"]}")
        if user_inputs["Days"] != "" and user_inputs["Substrate"] != "":
            print("days & substrate")
            path = [[None for _ in range(len(user_inputs["Days"]))] for _ in range(len(user_inputs["Substrate"]))]
            filename_pattern = [[None for _ in range(len(user_inputs["Days"]))] for _ in range(len(user_inputs["Substrate"]))]
            for s in range(len(user_inputs["Substrate"])):
                for d in range(len(user_inputs["Days"])):
                    path[s][d] = user_inputs["Path"] + path_delimiter + user_inputs["Substrate"][s] + path_delimiter + user_inputs["Days"][d] #reconstruct path
                    if os.path.isdir(path[s][d]): #If path exists reconsutruct filename pattern
                        pat = user_inputs["Filenames"].replace('Day', user_inputs["Days"][d])
                        pat = pat.replace('Substrate', user_inputs["Substrate"][s])
                        filename_pattern[s][d] = pat
                    else:
                        path[s][d] = None
                        filename_pattern[s][d] = None
        elif user_inputs["Days"] == "" and user_inputs["Substrate"] == "":
            print("no days and substrate")
            path = [[None]] ; filename_pattern = [[None]]
            path[0][0] = user_inputs["Path"]
            filename_pattern[0][0] = user_inputs["Filenames"]
            user_inputs["Days"] = [None] ; user_inputs["Substrate"] = [None]
            user_inputs["Days"][0] = "" ; user_inputs["Substrate"][0] = ""

        print('Path and filename pattern have same number of elements:', len(path) == len(filename_pattern))
    
    else:
        path = [[None]] ; filename_pattern = [[None]]
        path[0][0] = user_inputs["Path"]
        filename_pattern[0][0] = user_inputs["Filenames"]
    
    file_name = [[[] for _ in range(len(user_inputs["Days"]))] for _ in range(len(user_inputs["Substrate"]))]
    rm_path = []
    print(file_name, path, filename_pattern)
    for s in range(len(user_inputs["Substrate"])):
        for d in range(len(user_inputs["Days"])):
            files = os.listdir(path[s][d]) #list all files from directory
            valid_extension = ['tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
            #Store all file that have valid extension (ie pictures) and contain in their name the file name pattern
            for f in files:
                if (any(ext in f for ext in valid_extension)) and (filename_pattern[s][d] in f): 
                    file_name[s][d].append(f)

            #Verify there is the good number of picture and sort them by alphabetical order
            if len(file_name[s][d]) > n_expect or len(file_name[s][d]) == 0:
                rm_path.append(path[s][d])
                file_name[s][d] = None
                path[s][d] = None
                filename_pattern[s][d] = None
            else:
                file_name[s][d].sort()
    print(len(user_inputs["Substrate"])*len(user_inputs["Days"]) - len(rm_path), "/", 
          len(user_inputs["Substrate"])*len(user_inputs["Days"]), "folders contain usable pictures.")
    if len(rm_path) > 0:
        print("The following path will not be used: ", rm_path)

    return file_name, filename_pattern, path

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
        print(missing_img, missing_idx)

        #Add empty image and file name at good location for missing pictures
        for i in missing_idx:
            out_images.insert(i, np.zeros(images[0].shape, dtype=images[0].dtype))
            out_file_name.insert(i, filename_pattern + "%04d" % i + '.jpeg')
    
    del(images, filename_pattern, n_expect)
    return out_images, out_file_name

def find_picture_angle(images, hive, angle_user, angle_range, angle_step, display='off'):
    """
    returns angle of picture rotation. this angle is the median of each image best angle.
    Each image best angle is selected as the angle for which the results of hive and the picture convolution contains the highest pixel value.
    
    Parameters: 
        images (list) of (ndarray): each element is a picture (2D, grayscale).
        hive (ndarray): binary image of an hexagon representing the hives.
        angle_user (int): rough angle entered by user.
        angle_range (int): best angle will be search in interval [angle_user - angle_range ; angle_user - angle_range]
        angle_step (float): best angles will be search for each value in range [angle_user - angle_range ; angle_user - angle_range] with a step of angle_step 

    Returns:
        a_best (int): angle of rotation of the picture.
    """
    if isinstance(images, np.ndarray):
        images = [images]

    best_angles = np.empty(len(images)) 
    for i in range(len(images)):

        print('Determining angle to have picture in good orientation:', i+1, '/', len(images), end='\r')
        angle_array = np.arange( -angle_range + angle_user, angle_range + angle_user, angle_step) #array containing all angles to test
        conv = [None]*len(angle_array) #list where each element will contain the convolution result
        v_max = [None]*len(angle_array) #list where each element will contain highest pixel value of the convolution result
        
        #Convolution at different angles
        j=0
        r=0; c=0
        for a in angle_array:
            conv[j] = fftconvolve(images[i], ski.transform.rotate(hive, a), mode='same') #convolution in fourrier space
            v_max[j] = conv[j].max().max() #store highest pixel value
            j+=1

        #Display for each image the different angles tested
        if display == 'on' and i == 4:
            fig, ax = plt.subplots(nrows=2, ncols=4, num=f"Finding best angle on image {i}")
            for j,a in enumerate([0, 3, 6, 9]):
                ax[0, j].imshow(ski.transform.rotate(hive, angle_array[a]))
                ax[1, j].imshow(conv[a])
                ax[0, j].set_title(f"{angle_array[a]}°")
                ax[1, j].set_title(f"{round(v_max[a], 3)}")
               
        #Find best angle
        i_best = v_max.index(max(v_max))
        a_best = angle_array[i_best]
        best_angles[i] = a_best
    
    print()

    #Display the best angles of each image
    if display == 'on':
        nrow,ncol = get_compact_grid(len(images))
        fig2, ax2 = plt.subplots(nrows=nrow, ncols=ncol, num=f"Finding rotation angle of the session")
        j=0
        for r in range(nrow):
            for c in range(ncol):
                if j < len(images):
                    ax2[r, c].imshow(images[j])
                    ax2[r, c].set_title(f"{best_angles[j]}")
                j+=1
    
    #Find best angle for rotation by taking the median of all best angles
    a_best = np.nanmedian(best_angles)
    print('Best angle is:', a_best)

    del(images, hive, angle_user, angle_range, angle_step)
    return a_best

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

def cut_value_for_crop(barycenter, crop_length, img_length):
    """
    returns coordinates for crop along one axis (either x or y). Note that the distance between the crop coordinates are always equal to 2*crop_lendth
    
    Parameters: 
        barycenter (int): barycenter value along one axis. Should be the same axis than other parameters.
        crop_length (int): image diemension after crop along one axis. Should be the same axis than other parameters.
        img_length (int): image dimension along one axis. Should be the same axis than other parameters.

    Returns:
        list of two (int) elements: 1st element is the starting value for crop and 2nd element is the ending value for crop along the axis given in parameter.
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

def gauss_fit(x, *params):
    """Apply gaussian function on x values using the params as parameters"""

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def custom_threshold(img, fig, ax, title, display='off'):
    """Return threshold values under which the background is removed:
            -Find the peak in highest pixel values (corresponds to background (ie: mesh))
            -Fit the peaks with Gaussian
            -Set the threshold values as x0-3*sigma of the gaussian fit"""

    print('Determination of threshold values . . .', end='\r')

    #Variable
    pxl_depth = 255

    # #Create figure that will display histogram, peaks and fits
    # if len(ax) == 0 and display == 'off':
    #     fig, ax = plt.subplots()

    #Get histogram with 255 bins in range 0:1
    if img.flatten().min() < 0:
        data = ax.hist(img.flatten(), bins=pxl_depth, color='gray')
    else:
        data = ax.hist(img.flatten(), bins=pxl_depth, range=[0,1], color='gray')
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
    otsu = round(ski.filters.threshold_otsu(img) * pxl_depth)

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
    print("Determination of threshold values done.")
    print("\tOtsu 2 groups thresh =", otsu)
    print(f"\tLow pixel intensity peak (x,y): {peak_low}; \tGaussian fit Params: {popt_low}; \tMin threshold defined as {low_cut_type}")
    print(f"\tHigh pixel intensity peak (x,y): {peak_high}; \tGaussian fit Params: {popt_high}; \tMin threshold defined as {high_cut_type}")
    print("\tThreshold value (in range [ 0|0 ; 1|255 ]: ", end=" ")
    [print(f"{elt}|{elt*pxl_depth}", end=" ") for elt in x_cut]
    print()

    #Display curves & annotate
    if display == 'on' and len(ax) > 0:
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

    #Don't show picture if asked by user
    if display == 'off':
        plt.close(fig)

    return x_cut, low_cut_type, high_cut_type

def on_click_discard_undiscard_hive(event, coords_hive=None, hive_idx=None, discarded_hives=None):
    """
    On click, discard/undiscard hive by 1) updating the discarded_hives output list and 2) visually by adding blue cross on the discarded hive.

    Parameters
    ----------
    event:
    coords_hive: list of tuple. Each element contains the tuple
    hive_idx:
    coords_bbox: list of tuple. Each list element contains the coordinates (tuple of float) of the bbox containing the organoid as (ymin, xmin, ymax, ymax)
    """
    
    if event.button is MouseButton.LEFT and event.inaxes: #Do smt only if click inside plot area

        print(f'Click at location {event.xdata}, {event.ydata}, {type(event.xdata)}')

        for i,elt in enumerate(coords_hive): #loop over the coordinates of hives

            if elt[0][0] <= event.xdata < elt[0][1] and elt[1][0] <= event.ydata < elt[1][1]: #Search which hive is selected

                print(f"Organoid bbox Element {i} in picture {hive_idx[i]}: {elt}")

                if len(discarded_hives) == 0 or hive_idx[i] not in discarded_hives: #if hive not in discarded hive list:
                        
                    event.inaxes.plot([elt[0][0], elt[0][1]], [elt[1][1], elt[1][0]], color='c') #Add the 1st line to form the cross
                    event.inaxes.plot([elt[0][1], elt[0][0]], [elt[1][1], elt[1][0]], color='c') #Add the 2nd line to form the cross
                    discarded_hives.append(hive_idx[i]) #Add hive number to the list of discarded hives

                elif hive_idx[i] in discarded_hives: #if hive already in discarded hive list:

                    discarded_hives.remove(hive_idx[i]) #remove hte hive from list

                    for ln in event.inaxes.lines: #loop over all the lines created in the figure
                        bbox = ln.get_bbox() #get the bounding box of each line

                        if bbox.x0 == elt[0][0] and bbox.x1 == elt[0][1] and bbox.y0 == elt[1][0] and bbox.y1 == elt[1][1]: #Check which lines have good coordinates
                            ln.remove() #remove the line
        
        event.canvas.draw_idle() #Update plot

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

def get_compact_grid(n_subplots):

    n_sqrt = round(sqrt(n_subplots))
    if n_subplots % n_sqrt == 0:
        return n_sqrt, n_subplots // n_sqrt
    elif n_sqrt*n_sqrt > n_subplots:
        return round(n_sqrt), round(n_sqrt)
    else:
        return round(n_sqrt), round(n_sqrt) + 1

def open_results(graph_user_inputs, path_delimiter):
    
    data = pd.DataFrame() #data = pd.DataFrame(data=None, index=None, columns=heads, dtype=heads_type)

    #Verify path exists:
    print("find Days and Substrate in graph_user_input keys")
    if graph_user_inputs["Days"] != "" and graph_user_inputs["Substrate"] != "":
        if len(graph_user_inputs["Condition"]) == len(graph_user_inputs["Substrate"]):
            cond = True
        else:
            cond = False
        for p in graph_user_inputs["Path"]:
            for s_i, s in enumerate(graph_user_inputs["Substrate"]):
                for d in graph_user_inputs["Days"]:
                    path = p + path_delimiter + s + path_delimiter + d #reconstruct path
                    if os.path.isdir(path):
                        tmp = pd.read_csv(path + path_delimiter + 'UpdatedResults.csv')
                        if cond == True:
                            tmp["Condition"] = graph_user_inputs["Condition"][s_i]
                        else:
                            tmp["Condition"] = None
                        # print(s, d, "\n", tmp)
                        data = pd.concat([data, tmp], ignore_index=True)

    elif graph_user_inputs["Days"] == "" and graph_user_inputs["Substrate"] == "":
        path = graph_user_inputs["Path"]
        if os.path.isdir(path):
            tmp = pd.read_csv(path + path_delimiter + 'UpdatedResults.csv')
            tmp["Condition"] = None
            data.append(tmp, ignore_index=True)
        
    return data

def get_all_attribute_names(obj):
    names = set(vars(obj).keys())  # instance attributes

    # Add property names
    for name, value in obj.__class__.__dict__.items():
        if isinstance(value, property):
            names.add(name)

    return sorted(names)

def add_group_counts(ax: plt.axes, data: pd.DataFrame, x: str, hue=None, y_offset=0.02, fontsize=15):
    """
    Add sample size (n=...) labels on top of each group in a Seaborn categorical plot.

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

def display_organoid_evlotuion(filenames, days, substrate):
    
    #Determine which axis is the smallest
    if len(days) >= len(substrate):
        cols = days
        rows = substrate
    else:
        cols = substrate
        rows = days
    
    #Create figure
    fig, ax = plt.subplots(rows=len(rows), cols=len(cols), sharex=True, sharey=True)
    fig.tight_layout(pad=0)

    for c_i, c in enumerate(cols):
        for r_i, r in enumerate(rows):
            if len(days) >= len(substrate):
                img = np.ndarray(ski.io.imread(filenames[c_i][r_i]))
            else:
                img = np.ndarray(ski.io.imread(filenames[r_i][c_i]))
            ax[r_i, c_i].imshow(img)
            if c_i == 0:
                ax[r_i, c_i].set_ylabel(r)
        ax[0, c_i].set_title(c)

    
    plt.show()


