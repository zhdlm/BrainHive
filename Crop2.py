import sys
import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
import os
from PIL import Image
import logging
from math import floor
import re
import multiprocessing as mp
from functools import partial

#############################################################################################
# Custom functions
#############################################################################################

from custom_functions import hexagon_outline_ndarray, read_usr_input, assign_param, store_img_filename, add_empty_img, find_picture_angle, adjust_dimension_after_rotation, cut_value_for_crop, create_montage

#Determine delimiter for path depending on computer OS
system = os.name
match system:
    case 'posix':
        path_delimiter = "/"
    case 'nt':
        path_delimiter = "\\"


#############################################################################################
# Based on user input, load data
#############################################################################################

#Read user inputs and store user inputs in diactionnary
usr_inputs = read_usr_input("inputs.txt")
#print(usr_inputs)

#Parameters depending on user inputs
h, n_expected, um2pxl = assign_param(usr_inputs)
side = round(h*um2pxl/(2*np.sin(np.pi/3)))
diag = 2*side
th = 5

#Store the name of the pictures in a list
filenames, filename_pattern, path = store_img_filename(usr_inputs, path_delimiter, n_expected)

#Usefull variable
n_substrate = len(usr_inputs["Substrate"])
n_days = len(usr_inputs["Days"])

#Rescale ratio
scale_ratio=10

#############################################################################################
# For each susbrate and each day entered in user inputs.txt
#############################################################################################
z=0
for s in range(n_substrate):
    for d in range(n_days):

        print('\nSet ', z, '/', n_substrate*n_days, ':', path[s][d])
        z = z+1
        if path[s][d] == None:
            continue

        ####################################################################################
        # Open and verify pictures
        ####################################################################################

        n_file = len(filenames[s][d])

        #Open pictures, resize it, put in grayscale and store it in a list
        img = [None]*n_file #list that will containg the raw pictures
        img_rs = [None]*n_file #list that will contained the rescaled and grayscale pictures
        img_rs_sz= [None]*n_file
        for i in range(n_file):
            img_full_path = path[s][d] + path_delimiter + filenames[s][d][i] #complete path to picture
            img[i] = np.array(ski.io.imread(img_full_path)) #open picture
            img_sz = img[i].shape #original image_dimension
            img_gray = ski.color.rgb2gray(img[i]) #convert to grayscale
            img_rs[i] = ski.transform.resize(img_gray, (round(img_sz[0]/scale_ratio), round(img_sz[1]/scale_ratio))) #rescale the grayscale picture
            img_rs_sz[i]=img_rs[i].shape #rescaled image dimension
            img_type = img[0].dtype #store initial pixel value type
            print('Opening image, adujsting to grayscale and rescaling to', scale_ratio, ': ', i+1, '/', n_file, end='\r')
        print()

        #Verify the shapes are consistent
        if len(set(img_rs_sz)) > 1:
            print("Pictures do not have same dimension:\n", img_rs_sz)
            quit()
        else:
            img_rs_sz = img_rs_sz[0]
            print('Pictures successfully oppened.' \
            '\nOriginal image dimension: ', img_sz, \
            '\nImage dimension after ', scale_ratio,' times rescale: ', img_rs_sz)

        ####################################################################################
        # Determine rotation angle of the picture
        ####################################################################################

        #Creation of a hive image for rescaled dimension
        h_conv = round(h*um2pxl/scale_ratio) #distance between two opposite sides of the hexagon
        s_conv = round(h_conv/(2*np.sin(np.pi/3))) #side of the hexagon
        d_conv = 2*s_conv #long diagonal of the hexagon
        print('Hive dimensions in pixel rescaled for microscope ', usr_inputs["Microscope"], 'at magnification ', 
            usr_inputs["Magnification"], ': height=', h_conv, ', side=', s_conv, ', diagonal=', d_conv)
        hive_rs = hexagon_outline_ndarray(img_rs_sz, [img_rs_sz[0]/2, img_rs_sz[1]/2], s_conv, thickness=th)

        #Determine good angle
        a_best = find_picture_angle(img_rs, hive_rs, usr_inputs["Angle"], 5, 0.5, display='off')

        ####################################################################################
        # Crop the picture around the hive
        ####################################################################################

        for i in range(len(img_rs)):
            print('Computing hexagon barycenter and cropping picture:', i+1, '/', len(img_rs), end='\r')           

            if sum(sum(img_rs[i])) != 0:

                #Pre-process of the picture
                blur = scipy.ndimage.gaussian_filter(img_rs[i], 5/scale_ratio) #blur pictire to smooth edges
                otsu_thresh = ski.filters.threshold_otsu(blur) #compute the otsu automatic threshold
                thresh = np.empty(blur.shape) 
                thresh[blur >= otsu_thresh] = 0 #value below thresh =0
                thresh[blur < otsu_thresh] = 1 #value above thresh =1
                rot = ski.transform.rotate(thresh, - a_best, preserve_range=True, resize=True) #rotate the blured and thresholded picture to have it in good orientation

            else:
                rot = ski.transform.rotate(img_rs[i], - a_best, preserve_range=True, resize=True)

            #Find hive on the picture
            conv = scipy.signal.fftconvolve(rot, hive_rs, mode='same') #perform convolution of hive mask and pre-processes imaged (blurred, thesrloded and rotated)
            # plt.figure('conv'+filenames[i])
            # plt.imshow(conv)
            tmp = conv.copy()
            v_max = tmp.max().max() #compute maximum pixel value on the results of convolution
            tmp[tmp < v_max] = 0
            # plt.figure('conv cut'+filenames[i])
            # plt.imshow(tmp)

            #Compute barycenter of the convolution image
            if sum(sum(img_rs[i])) != 0:
                y_c = sum(tmp.sum(axis=1)*range(tmp.shape[0]))/sum(tmp.sum(axis=1)) #projection on y axis and computing barycenter of y
                x_c = sum(tmp.sum(axis=0)*range(tmp.shape[1]))/sum(tmp.sum(axis=0)) #projection on x axis and computing barycenter of x
            else:
                y_c = 0
                x_c = 0

            #Rotate original image to have it in good orientation
            img[i] = ski.transform.rotate(img[i], - a_best, preserve_range=True, resize=True)
            img[i] = img[i].astype(img_type)
            #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)
            img[i] = adjust_dimension_after_rotation(img[i], [round(h*um2pxl), round(diag)], img_type)

            #Convert the barycenter at good scale
            y_c = round(img[i].shape[0]*y_c/conv.shape[0])
            x_c = round(img[i].shape[1]*x_c/conv.shape[1])

            # print('Coordinate of barycenter (x,y): (', y_c, ',', x_c, ')')
            #del(conv, tmp)

            #Create hive mask for the crop with center as barycenter of convolution picture
            hive_mask =  hexagon_outline_ndarray(img[i].shape, [y_c, x_c], side, thickness=th) #recreate the hexagon outlines at good dimension
            hive = ski.segmentation.flood_fill(hive_mask, (y_c, x_c), 1) #fill the hexagon
            hive = hive.astype(int)
            #Force inner part to be set to at 1
            if hive[y_c, x_c] == 0:
                hive = np.invert(hive)
            # plt.figure("label"+str(i))
            # plt.imshow(label)
            # plt.colorbar()
            # plt.show()

            #Compute x and y values where crop should occure:
            y_cut = cut_value_for_crop(y_c, h*um2pxl/2, img[i].shape[0])
            x_cut = cut_value_for_crop(x_c, diag/2, img[i].shape[1])

            #Crop the picture
            img[i][hive == 0] = 0 #assign all values outside hive to 0
            img[i] = img[i][y_cut[0]:y_cut[1], x_cut[0]:x_cut[1], :] #crop around h and diag
            # plt.figure('Image cropped'+filenames[s][d][i])
            # plt.imshow(img[i])

            out_path = path[s][d] + path_delimiter + "Crop_" + filenames[s][d][i]
            ski.io.imsave(out_path, img[i])

            # if i==4:
            #     fig, ax = plt.subplots(nrows=3, ncols=3)
            #     ax[0, 0].imshow(img_rs[i])
            #     ax[0, 0].set_title("Grayscale image")
            #     ax[0, 1].imshow(img_rs[i])
            #     ax[0, 1].set_title("Gaussian filter")
            #     ax[0, 2].imshow(thresh)
            #     ax[0, 2].set_title("Thesrholded")
            #     ax[1, 0].imshow(rot)
            #     ax[1, 0].set_title("Rotated")
            #     ax[1, 1].imshow(hive_rs)
            #     ax[1, 1].set_title("Hexagon mask")
            #     ax[1, 2].imshow(conv)
            #     ax[1, 2].set_title("Convolution output")
            #     ax[2, 0].imshow(tmp)
            #     ax[2, 0].set_title("Highest value only")
            #     ax[2, 1].imshow(rot)
            #     ax[2, 1].set_title("Barycenter")
            #     ax[2, 1].scatter(round(x_c/scale_ratio), round(y_c/scale_ratio), marker="x", color="r")
            #     ax[2, 2].imshow(img[i])
            #     ax[2, 2].set_title("Crop")

        ####################################################################################
        # Montage of entire subsrate
        ####################################################################################
        
        #Create Montage image
        montage = create_montage(usr_inputs, img, img_type, n_expected, filenames[s][d])
        # plt.figure('Montage')
        # plt.imshow(montage)
        print()
        plt.show()

        print(usr_inputs["Path"], path_delimiter, 'Montage_', usr_inputs["Days"][d], '_', usr_inputs["Substrate"][s], str(a_best), 'degree.jpeg')
        #Save Montage image
        try:
            #full resolution picture
            out_path = usr_inputs["Path"] + path_delimiter + 'Montage_' + usr_inputs["Days"][d] + '_' + usr_inputs["Substrate"][s] + str(a_best) + 'degree.jpeg'
            print(out_path)
            ski.io.imsave(out_path, montage)
            print('Imaged saved as: ', out_path)

            #low resolution picture
            out_path = usr_inputs["Path"] + path_delimiter + 'Montage_' + usr_inputs["Days"][d] + '_' + usr_inputs["Substrate"][s] + 'lowQ.jpeg'
            tmp = ski.transform.downscale_local_mean(montage, (6, 6, 1)) #much quicker than resize and sufficient for low quatlity picture in reports
            tmp = tmp.astype(img_type)
            ski.io.imsave(out_path, tmp) #force variable type to enable saving picture
            print('Imaged saved as: ', out_path)
            del(out_path)

        except:
            print('Image could not be saved in ', out_path)