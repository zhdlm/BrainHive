import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
import os
from datetime import datetime
from PIL import Image
import csv
import itertools
from functools import partial
import re


#############################################################################################
# Custom functions
#############################################################################################

from custom_functions import read_usr_input, assign_param, store_img_filename, hexagon_outline_ndarray, fine_tuning_crop, custom_threshold, gauss_fit, get_compact_grid, create_montage, get_hive_coord_montage, on_click_discard_undiscard_hive, on_click_discard_undiscard_organoid, get_all_attribute_names


#############################################################################################
# Based on user input, load data
#############################################################################################

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

#Properties of interest
props = ['label', 'centroidX', 'centroidY', 'centroid_localX', 'centroid_localY', 'bbox_ymin', 'bbox_xmin', 'bbox_ymax', 'bbox_xmax', 
         'eccentricity', 'area_filled', 'perimeter', 'equivalent_diameter_area', 'axis_major_length', 'axis_minor_length',
         'intensity_mean', 'intensity_std',
         'image_filled', 'image_intensity']
headers = props + ['Index', 'Day', 'Substrate', 'Image Name', 'Hive number', 'Manually Discarded','Analysis Date', 'Analysis Version']
# headers.remove('centroid') ; headers.remove('centroid_local')
# headers += ['centroid-0', 'centroid-1', 'centroid_local-0', 'centroid_local-1']

#Read user inputs and store user inputs in diactionnary
usr_inputs = read_usr_input("inputs.txt")
usr_inputs["Filenames"] = "Crop_" + usr_inputs["Filenames"]
print(usr_inputs)

#Parameters depending on user inputs
h, n_expected, um2pxl = assign_param(usr_inputs)
mesh_pxl = round(60*um2pxl / 2)

#Store the name of the pictures in a list
#filenames = store_img_filename(usr_inputs, "Crop_", n_expected) #only the cropped pictures
# filenames = store_img_filename(usr_inputs, path_delimiter, n_expected) #for test data set
files = os.listdir(usr_inputs["Path"])

#filenames = [f for f in files if os.path.isfile(usr_inputs["Path"] + path_delimiter + f)]
filenames, filename_pattern, paths = store_img_filename(usr_inputs, path_delimiter, n_expected)

#############################################################################################
# Optimize crop & segmentation
#############################################################################################

#Initialize output variable
masks = [[[] for _ in range(len(usr_inputs["Days"]))] for _ in range(len(usr_inputs["Substrate"]))]
img = [[[] for _ in range(len(usr_inputs["Days"]))] for _ in range(len(usr_inputs["Substrate"]))]
data = [[[] for _ in range(len(usr_inputs["Days"]))] for _ in range(len(usr_inputs["Substrate"]))]


#For each substrate & each imaging session
for s in range(len(usr_inputs["Substrate"])):
    for d in range(len(usr_inputs["Days"])):
        
        #Go to next folder if this one is empty
        if paths[s][d] == None:
            continue

        #Open csv file to store segmentation results
        f = open(paths[s][d] + path_delimiter + "Results.csv", 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        #Get shape and type of original pictures
        img_full_path = paths[s][d] + path_delimiter + filenames[s][d][0]
        tmp_img = np.array(ski.io.imread(img_full_path))
        img_shape = tmp_img.shape
        img_dtype = tmp_img.dtype
        #Get shape and type of grayscale image
        tmp_gray_img = ski.color.rgb2gray(tmp_img)
        gray_img_shape = tmp_gray_img.shape
        gray_img_dtype = tmp_gray_img.dtype
        #Continue initialization of output variables
        img[s][d] = [np.zeros(img_shape, dtype=img_dtype) for _ in range(len(filenames[s][d]))] #[np.zeros(img_shape, dtype=img_dtype)]*len(filenames[s][d])
        data[s][d] = [[] for _ in range(len(filenames[s][d]))]
        masks[s][d] = [np.zeros(gray_img_shape, dtype=gray_img_dtype) for _ in range(len(filenames[s][d]))] #[np.zeros(gray_img_shape, dtype=gray_img_dtype)]*len(filenames[s][d])
        overlays = [np.zeros((gray_img_shape[0], gray_img_shape[1], 3), dtype=np.uint8) for _ in range(len(filenames[s][d]))] #will hold overlays of the session
        del(img_full_path, tmp_img, tmp_gray_img)

        #Slice of hexagon of thickness = k_step
        n_slice = 15
        hives_mask_side = [None] * n_slice
        side = round(h*um2pxl/(2*np.sin(np.pi/3))) #0.94 factor to do an hexagon that will not contain the shadow from the walls of the hive
        height = h*um2pxl 
        y_c = round(img_shape[0]/2)
        x_c = round(img_shape[1]/2)
        k_step = 10
        # fig1,ax1=plt.subplots(ncols=n_slice)
        for j,k in enumerate(range(k_step, n_slice*k_step+k_step, k_step)):
            side_k = round((height - 2*k)/(2*np.sin(np.pi/3)))
            hives_mask_side[j] = hexagon_outline_ndarray(img_shape, [y_c, x_c], side_k, thickness=k_step, return_type='side')
        idx_test = [0, 3, 9, 11, 17, 21, 18, 31, 33]

        #Figure to display the different steps
        n_r, n_c = get_compact_grid(len(filenames[s][d]))
        fig2, ax2 = plt.subplots(nrows=n_r, ncols=n_c, sharex=True, sharey=True)
        fig, ax = plt.subplots(nrows=8, ncols=len(filenames[s][d]), sharex=True, sharey=True)
        fig2.supxlabel('Pixel value (in range [0;1])')
        fig2.supylabel('Counts')
        if n_r > 1:
            idx_ax2 = list(itertools.product(range(n_r), range(n_c)))
        else:
            idx_ax2 = [elt for elt in range(n_c)]

        #Will hold organoid counts per imaging session
        organoid_count = 0

        fig, ax = plt.subplots(nrows=4, ncols=len(filenames[s][d]))

        #j=0
        #For each hive (ie each image)
        for i in range(len(filenames[s][d])):

            j=i
            idx_current_ax = idx_ax2[i]

            print(f"\n{i+1}/{len(filenames[s][d])}: Image {filenames[s][d][i]}")

            #Open the pictures
            img_full_path = paths[s][d] + path_delimiter + filenames[s][d][i]
            img[s][d][i] = np.array(ski.io.imread(img_full_path)) #cropped picture
            img_gray = ski.color.rgb2gray(img[s][d][i]) #cropped picture in gray scale
            hive_slice_mask = [np.zeros(img_gray.shape) for _ in range(6)]
            organoid_mask = np.zeros(img_gray.shape)

            #Apply gaussian blur
            r_blur=5*um2pxl #to blur unique cells entierly (considered cells about 10µm diameter)
            img_blur = scipy.ndimage.gaussian_filter(img_gray, r_blur)

            #Fine tuning of hexagon mask to get rid of hive borders & shadows of hive walls
            hive_mask = fine_tuning_crop(img_gray, hives_mask_side, height, k_step, n_slice, display='on')
                
            #Force inner part to be set to at 1
            hive_mask = ski.morphology.flood_fill(hive_mask, (x_c, y_c), 1)
            hive_mask = hive_mask.astype(int)
            if hive_mask[y_c, x_c] == 0:
                hive_mask = np.invert(hive_mask)

            #Save the Optimized cropped picture
            img[s][d][i][hive_mask == 0] = 0
            out_path = paths[s][d] + path_delimiter + filenames[s][d][i]
            out_path = out_path.replace('Crop', 'OptimizedShadow')
            ski.io.imsave(out_path, img[s][d][i])

            #Remove the areas nearby the walls to remove artefacts
            img_gray[hive_mask == 0] = 0
            img_blur = scipy.ndimage.gaussian_filter(img_gray, r_blur)

            #Find the interval for threshold
            thresh_value, low_thresh_type, high_thresh_type = custom_threshold(img_blur, fig2, ax2[idx_current_ax], filenames[s][d][i], display='off')
            #Apply threshold
            img_thresh = (img_blur > thresh_value[0]) & (img_blur <= thresh_value[1])
            ax[1, j].imshow(img_thresh)
            

            #Close areas
            img_thresh = ski.morphology.binary_erosion(img_thresh, footprint=ski.morphology.ellipse(mesh_pxl, mesh_pxl)) #to get rid of the segmented mesh
            img_thresh = ski.morphology.area_closing(img_thresh)
            img_thresh = ski.morphology.binary_dilation(img_thresh, footprint=ski.morphology.ellipse(mesh_pxl, mesh_pxl))
            ax[2, j].imshow(img_thresh)
 

            #Find the different regions, label it
            label, num = ski.measure.label(img_thresh, return_num=True, connectivity=2)
            regions = ski.measure.regionprops(label, intensity_image=img_gray)
            
            # Find regions not satisfiying morphological criteria:
            # eccentricity <0.8
            # area in between 0.007-1.5 mm²
            labs = []
            for rg in regions:
                if rg.area > 30000:
                    print(rg.label, rg.eccentricity, rg.area_filled)
                #Find ones to remove
                if (rg.eccentricity > 0.88) or (rg.area_filled <= 30000) or (rg.area_filled >= 2000000):
                   labs.append(rg.label)
            #Remove them
            for k in sorted(labs, reverse=True):
                #print("Remove:", len(regions), labs, k)
                del(regions[k-1])
                label[(label == k)] = 0
            print(f"{np.unique(label)} --> {len(np.unique(label)) - 1} organoids found.")
            #Display the filtered regions
            #ax[6, j].imshow(label)

            #Fill the holes in accurate regions
            for rg in regions:
                r_min, c_min, r_max, c_max = rg.bbox
                label[r_min:r_max, c_min:c_max] = rg.image_filled

            #Display different steps
            ax[0, j].imshow(img_gray)
            ax[3, j].imshow(label)

            #Binarize mask
            organoid_mask = label.copy()
            organoid_mask[label == 0] = False
            organoid_mask[label > 0] = True
            # #Dilate a bit the mask to take borders
            # organoid_mask = ski.morphology.binary_dilation(organoid_mask, footprint=ski.morphology.ellipse(6,6))
            #Mask from 0 to 255 to be able to save it in jpeg format
            organoid_mask = organoid_mask.astype(np.uint8)
            organoid_mask[organoid_mask > 0] = 255
            #Copy mask into output variable
            masks[s][d][i] = organoid_mask
            #Display the final mask
            #ax[7, j].imshow(masks[s][d][i])
            # ax[7, j].imshow(ski.feature.canny(img_blur, sigma=3))

            #Save mask
            out_path = paths[s][d] + path_delimiter + filenames[s][d][i]
            #out_path = paths[s][d] + path_delimiter + "Mask_" + filenames[s][d][i]
            out_path = out_path.replace('Crop', 'Mask')
            ski.io.imsave(out_path, masks[s][d][i])

            #Save overlay of raw picture + organoid mask (in red)
            overlay = np.zeros((gray_img_shape[0], gray_img_shape[1], 3), dtype=np.uint8)
            overlay[:,:,0] = img[s][d][i][:,:,0] + masks[s][d][i]*0.2 #R
            overlay[:,:,1] = img[s][d][i][:,:,1] #G
            overlay[:,:,2] = img[s][d][i][:,:,2] #B
            out_path = paths[s][d] + path_delimiter + filenames[s][d][i]
            #out_path = paths[s][d] + path_delimiter + "Overlay_" + filenames[s][d][i]
            out_path = out_path.replace('Crop', 'Overlay')
            ski.io.imsave(out_path, overlay)

            #Extract info on organoids & save it to Results.csv file
            for rg in regions:
                rg_props = get_all_attribute_names(rg)
                data[s][d][i].append({})
                [data[s][d][i][-1].update({head: rg[head]}) for head in props if head in rg_props] #from regionprops
                data[s][d][i][-1].update({"centroidX": float(rg.centroid[0]), "centroidY": float(rg.centroid[1]), 
                                          "centroid_localX": rg.centroid_local[0], "centroid_localY": rg.centroid_local[1]})
                data[s][d][i][-1].update({"bbox_ymin": float(rg.bbox[0]), "bbox_xmin": float(rg.bbox[1]), 
                                          "bbox_ymax": rg.bbox[2], "bbox_xmax": rg.bbox[3]})
                data[s][d][i][-1].update({"Index": organoid_count, "Day": usr_inputs["Days"][d], "Substrate": usr_inputs["Substrate"][s], 
                                          "Image Name": filenames[s][d][i], "Hive number": int(re.search('00[0-9][0-9]', filenames[s][d][i]).group()),
                                          "Manually Discarded": False, "Analysis Date": datetime.now(), "Analysis Version": "v0.1"}) #metadata
                organoid_count += 1 #incremente organoid count
                
            writer.writerows(data[s][d][i])
                 
            #Store the overlay
            overlays[i] = overlay

            # if i == 2:
            #     f.close()
            #     plt.show()
            #     quit()
            j+=1
        f.close()

        #Create montage of the overlays
        overlay_montage = create_montage(usr_inputs, overlays, overlays[i].dtype, n_expected, filenames[s][d])

        #Save overlay
        out_path = paths[s][d] + path_delimiter + 'Overlay_Montage_' + usr_inputs["Days"][d] + '_' + usr_inputs["Substrate"][s] + '.jpeg'
        tmp = ski.transform.downscale_local_mean(overlay_montage, (6, 6, 1)) #much quicker than resize and sufficient for low quatlity picture in reports
        tmp = tmp.astype(overlay_montage.dtype)
        ski.io.imsave(out_path, tmp) #force variable type to enable saving picture
        print('Imaged saved as: ', out_path)

        #Not to show the plots
        # plt.close(fig)
        plt.close(fig2)

plt.show()

#############################################################################################
# Manual discard of organoids
#############################################################################################


#Discard hives
for s in range(len(usr_inputs["Substrate"])):
    for d in range(len(usr_inputs["Days"])):

        if paths[s][d] == None:
            continue

        #Open overlay image of the substrate s and dession d
        overlay_full_path = paths[s][d] + path_delimiter + 'Overlay_Montage_' + usr_inputs["Days"][d] + '_' + usr_inputs["Substrate"][s] + '.jpeg'
        overlay_montage = np.array(ski.io.imread(overlay_full_path))

        #Image acquisition order:
        match usr_inputs["Acquisition order"]:
            case 'bottom':
                idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]]
            case 'bottom-flip':
                idx = [[16, 15, 7], [17, 14, 8, 6], [18, 13, 9, 5, 0], [12, 10, 4, 1], [11, 3, 2]]
            case 'bottom right':
                idx = [[2,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
            case 'exp001':
                idx = [[7,6,5], [15,8,4,0], [16,14,9,3,1], [17,13,10,2], [18,12,11]]
            case 'bottom center':
                # idx = [[,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
                pass
        
        #Get hive coordinates
        coord_hives, idx_flatten = get_hive_coord_montage(overlay_montage.shape, idx, base='montage')
        #print(coord_hives, tmp, idx)


        #Reorganize coordinates into list of dictionnary
        coords = []
        for i in range(len(data[s][d])):
            for j in range(len(data[s][d][i])):
                coords.append({"Index": data[s][d][i][j]["Index"], \
                               "Hive number": data[s][d][i][j]["Hive number"], \
                               "Hive coord": coord_hives[idx_flatten.index(data[s][d][i][j]["Hive number"])],
                               "bbox": (data[s][d][i][j]["bbox_ymin"]/6, data[s][d][i][j]["bbox_xmin"]/6, data[s][d][i][j]["bbox_ymax"]/6, data[s][d][i][j]["bbox_xmax"]/6) #coordinates of resize image, division factor of 6 to have the rescale one
                               })     

        #Create figure
        fig3, ax3 = plt.subplots()
        ax3.imshow(overlay_montage)
        discarded_organoids = [] #will hold the discarded hives
        handler_on_click = partial(on_click_discard_undiscard_organoid, coords=coords, discarded_organoids=discarded_organoids) #to enable passing variable to plt.connect functions
        fig3.canvas.mpl_connect('button_press_event', handler_on_click) #callbaks to discard/undiscard hives when selecting a hive on the picture
        plt.show()

        #Once picture is closed, update data & update results file
        f = open(paths[s][d] + path_delimiter + "UpdatedResults.csv", 'w+', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        if len(discarded_organoids) > 0:
            for i in range(len(data[s][d])): #loop over each picture (ie hive)
                #print(data[s][d][i], type(data[s][d][i]), len(data[s][d][i]))
                if len(data[s][d][i]) > 0:
                    for j in range(len(data[s][d][i])): #loop over each object from the picture (ie hive)
                        if data[s][d][i][j]["Index"] in discarded_organoids:
                            data[s][d][i][j]["Manually Discarded"] = True
                        else:
                            data[s][d][i][j]["Manually Discarded"] = False
                    writer.writerows(data[s][d][i])
        else: #simply copy the data in UpdatedResults.csv
            for i in range(len(data[s][d])):
                writer.writerows(data[s][d][i])
        f.close()
        print(f"The discarded hives are: {discarded_organoids}")