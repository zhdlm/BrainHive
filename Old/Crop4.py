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
from math import sqrt

#############################################################################################
# Custom functions
#############################################################################################

from custom_functions import hexagon_outline_ndarray, read_usr_input, assign_param, store_img_filename, add_empty_img, find_picture_angle, adjust_dimension_after_rotation, cut_value_for_crop

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
        binary_rs = [None]*n_file #list that will contained the rescaled and binary picture of hive
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
        # Find hive
        ####################################################################################

        #Plots:
        fig, ax = plt.subplots(nrows=5, ncols=len(img_rs))
        for i in range(len(img_rs)):


            ax[0, i].imshow(img_rs[i])

            #Blur
            blur = ski.filters.gaussian(img_rs[i], 3)
            ax[1, i].imshow(blur)

            #Thresh to select hive
            otsu = ski.filters.threshold_otsu(blur)
            thresh = blur > otsu
            ax[2, i].imshow(thresh)

            #Keep biggest region
            label = ski.measure.label(thresh)
            regions = ski.measure.regionprops(label)
            biggest_region = None
            for rg in regions:
                if rg.label == 1:
                    biggest_region = rg
                elif rg.area_filled > biggest_region.area_filled:
                    biggest_region = rg
            label[label != biggest_region.label] = 0
            r_min, c_min, r_max, c_max = biggest_region.bbox
            label[r_min:r_max, c_min:c_max] = biggest_region.image_filled & 1

            ax[3, i].imshow(label)

            canny = ski.feature.canny(label)
            ax[4, i].imshow(canny)

            binary_rs[i] = label
            
            # closed = ski.morphology.binary_dilation(borders, footprint=ski.morphology.ellipse(3, 3)) #footprint=np.ones((5,5)
            # ax[3, i].imshow(closed)
            # closed = ski.morphology.binary_closing(closed, footprint=ski.morphology.ellipse(2, 2))
            # ax[4, i].imshow(closed)
            # # eroded = ski.morphology.binary_erosion(closed,footprint=ski.morphology.ellipse(2, 2))
            # # ax[5, i].imshow(eroded)
            # #label
            # label = ski.measure.label(closed)
            # ax[6, i].imshow(label)

            # h_conv = round(h*um2pxl/scale_ratio) #distance between two opposite sides of the hexagon
            # s_conv = round(h_conv/(2*np.sin(np.pi/3))) #side of the hexagon
            # regions = ski.measure.regionprops(label)
            # to_rm = []
            # for rg in regions:
            #     if rg.area_filled < 6*s_conv*s_conv*sqrt(3)/4*0.3:
            #         #print(f"{rg.label} to remove: {rg.area_filled}")
            #         to_rm.append(rg.label)
            # for r in sorted(to_rm, reverse=True):
            #     #print(r)
            #     label[label == r] = 0
            #     del(regions[r-1])
            # del(regions)

            # if len(np.unique(label)) <=1:
            #     label = closed
            # ax[7, i].imshow(label)

            # lines = ski.transform.probabilistic_hough_line(label, line_length=s_conv, line_gap=round(s_conv/3))
            # for li in lines:
            #    ax[8, i].axline([li[0][0], li[1][0]], [li[0][1], li[1][1]])

            # binary_rs[i] = label

            # #Find line using Hough Transform
            # lines = ski.transform.probabilistic_hough_line(borders, line_length=s_conv, line_gap=round(s_conv/3))
            # hspace, thetas, dists = ski.transform.hough_line(borders)
            
            # # #Draw lines
            # #hspace, thetas, dists = ski.transform.hough_line_peaks(hspace, thetas, dists)

            # for _, th, di in zip(*ski.transform.hough_line_peaks(hspace, thetas, dists)):
            #     (x0, y0) = di * np.array([np.cos(th), np.sin(th)])
            #     ax[2].axline((x0, y0), slope=np.tan(th + np.pi / 2))
            # for li in lines:
            #     ax[4].axline([li[0][0], li[1][0]], [li[0][1], li[1][1]])

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
        #hive_filled = ski.segmentation.flood_fill(hive, (y_c, x_c), 0) #fill 0 in the hexagon
        kernel = [[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1] ]

        #Determine good angle
        a_best = find_picture_angle(binary_rs, hive_rs, usr_inputs["Angle"], 5, 0.5)

        ####################################################################################
        # Crop the picture around the hive
        ####################################################################################

        for i in range(len(img_rs)):
            print('Computing hexagon barycenter and cropping picture:', i+1, '/', len(img_rs), end='\r')

            # if sum(sum(img_rs[i])) != 0:

            #     #Pre-process of the picture
            #     blur = scipy.ndimage.gaussian_filter(img_rs[i], 5/scale_ratio) #blur pictire to smooth edges
            #     otsu_thresh = ski.filters.threshold_otsu(blur) #compute the otsu automatic threshold
            #     thresh = np.empty(blur.shape) 
            #     thresh[blur >= otsu_thresh] = 0 #value below thresh =0
            #     thresh[blur < otsu_thresh] = 1 #value above thresh =1
            #     rot = ski.transform.rotate(thresh, - a_best, preserve_range=True, resize=True) #rotate the blured and thresholded picture to have it in good orientation

            # else:
            #     rot = ski.transform.rotate(img_rs[i], - a_best, preserve_range=True, resize=True)
            rot = ski.transform.rotate(binary_rs[i], - a_best, preserve_range=True, resize=True)

            #Find hive on the picture
            conv = scipy.signal.fftconvolve(rot, hive_rs, mode='same') #perform convolution of hive mask and pre-processes imaged (blurred, thesrloded and rotated)
            plt.figure('conv'+filenames[s][d][i])
            plt.imshow(conv)
            tmp = conv
            v_max = tmp.max().max() #compute maximum pixel value on the results of convolution
            tmp[tmp > 1] = 0
            tmp[0:round(img_rs_sz[0]/4), :] = 0
            tmp[round(3*img_rs_sz[0]/4):-1, :] = 0
            tmp[:, 0:round(img_rs_sz[1]/4)] = 0
            tmp[:, round(3*img_rs_sz[1]/4):-1] = 0
            plt.figure('conv cut'+filenames[s][d][i])
            plt.imshow(tmp)

            #Compute barycenter of the convolution image
            if sum(sum(img_rs[i])) != 0:
                y_c = sum(tmp.sum(axis=1)*range(tmp.shape[0]))/sum(tmp.sum(axis=1)) #projection on y axis and computing barycenter of y
                x_c = sum(tmp.sum(axis=0)*range(tmp.shape[1]))/sum(tmp.sum(axis=0)) #projection on x axis and computing barycenter of x
            else:
                y_c = 0
                x_c = 0
            print(y_c, x_c)

            #Rotate original image to have it in good orientation
            img[i] = ski.transform.rotate(img[i], - a_best, preserve_range=True, resize=True)
            img[i] = img[i].astype(img_type)
            #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)
            img[i] = adjust_dimension_after_rotation(img[i], [round(h*um2pxl), round(diag)], img_type)

            #Convert the barycenter at good scale
            y_c = round(img[i].shape[0]*y_c/conv.shape[0])
            x_c = round(img[i].shape[1]*x_c/conv.shape[1])

            # print('Coordinate of barycenter (x,y): (', y_c, ',', x_c, ')')
            del(conv, tmp)

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

        ####################################################################################
        # Montage of entire subsrate
        ####################################################################################

        # Substrate is definied as follow:
        # 1st column: 3 hives
        # 2nd column: 4 hives
        # 3rd column: 5 hives
        # 4th column: 3 hives
        # 5th column: 4 hives

        #Image acquisition order:
        match usr_inputs["Acquisition order"]:
            case 'bottom':
                idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]]
            case 'bottom right':
                idx = [[2,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
            case 'bottom center':
                # idx = [[,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
                pass

        idx_flat = [item for sublist in idx for item in sublist]
        print('Image acquisition order (', usr_inputs["Acquisition order"], '): ', idx_flat)

        #If missing pictures, adding empty ones
        img, tmp = add_empty_img(img, filenames[s][d], usr_inputs["Filenames"], n_expected)
        img_rs, filenames[s][d] = add_empty_img(img_rs, filenames[s][d], usr_inputs["Filenames"], n_expected)
        del(tmp)

        #Reconstruct final image
        img_sz = img[0].shape
        print('Cropped Images dimension (y,x):', img_sz)
        montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=img_type)
        i=0 #counting pictures in following nested loops
        r=0 #row on the substrate
        for c in [0,1,2,1,0]:
            #Initiate location of the first picture of the row
            y_min = round((5*img_sz[0] - (c+3)*img_sz[0])/2)
            y_max = round(y_min + img_sz[0])
            x_min = img_sz[1]*(r)
            x_max = x_min + img_sz[1]

            for elt in range(len(idx[r])):
                #print('c=', c, 'elt=', elt, 'coord= ', y_min, y_max, x_min, x_max)
                print('Creating monatge picture:', i+1, '/', len(img_rs), end='\r')
                #try :
                montage[y_min:y_max, x_min:x_max, :] = img[idx_flat[i]] #add picture to montage at good location
                # except:
                #     montage[y_min:y_max, x_min:x_max, :] = 0
                #Set paramaters for next picture from the row
                i += 1
                y_min += img_sz[0]
                y_max = y_min + img_sz[0]
                # plt.figure('Montage')
                # plt.imshow(montage)
                # plt.show()
            r+=1
        # plt.figure('Montage')
        # plt.imshow(montage)
        print()
        plt.show()

        try:
            #full resolution picture
            out_path = usr_inputs["Path"] + path_delimiter + 'Montage_' + usr_inputs["Days"][d] + '_' + usr_inputs["Substrate"][s] + str(a_best) + 'degree.jpeg'
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