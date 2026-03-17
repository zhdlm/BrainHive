import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
import os
from PIL import Image
import csv as csv
from enum import Enum
import itertools

#############################################################################################
# Custom functions
#############################################################################################

from custom_functions import read_usr_input, assign_param, store_img_filename, custom_threshold, hexagon_outline_ndarray

class BkgdRemoval(Enum):
    NONE = 0
    DIVISION = 1
    SUBTRACTION = 2

class RadiiBkgdRemoval(Enum):
    R_25 = 25
    R_50 = 50
    R_250 = 250
    R_500 = 500

class ThreshMethod(Enum):
    MEAN = "Mean"
    YEN ="Yen"
    OTSU = "Otsu"
    MULTI_OTSU_3 = "Multi-Otsu3"
    CUSTOM = "Custom"

class EllipseRadius(Enum):
    PXL_2 = 2
    PXL_4 = 4
    PXL_6 = 6
    PXL_8 = 8
    # PXL_10 = 10
    # PXL_20 = 20
    # PXL_MESH = round(60*0.9/2) #0.9 for um2pxl

#############################################################################################
# Based on user input, load data
#############################################################################################

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

#Read user inputs and store user inputs in diactionnary
usr_inputs = read_usr_input("inputs.txt")

#Parameters depending on user inputs
h, n_expected, um2pxl = assign_param(usr_inputs)

outpath = "D:\\Data\\DataSet_test\\randomSet-Analysis\\Segmentation Performances\\D:\Data\DataSet_test\randomSet-Analysis\Segmentation Performances\SegmentationPerf_OptimizedShadow_BkgdRemoval_ThreshGlobal_HiveMask_Erosion-Dilation2to8_Filter_FillHoles"
params = {"Apply hive mask ": True,
          "Erosion": True,
          "Dilation": True,
          "Filter": {"Status": True, "eccentricity": 0.88, "area_min": 30000, "area_max": 2000000},
          "Fill Holes": True}

#Store the name of the pictures in a list
#filenames = store_img_filename(usr_inputs, "Crop_", n_expected) #only the cropped pictures
# filenames = store_img_filename(usr_inputs, path_delimiter, n_expected) #for test data set
files = os.listdir(usr_inputs["Path"])
filenames = [f for f in files if os.path.isfile(usr_inputs["Path"] + path_delimiter + f)]
if "Results.csv" in filenames:
    filenames.remove("Results.csv")
n_file = len(filenames)

hive_mask = hexagon_outline_ndarray((1800, 2078), [1800/2, 2078/2], round(h*um2pxl/(2*np.sin(np.pi/3))), thickness=5, return_type='hexagon')
hive_mask = ski.segmentation.flood_fill(hive_mask, (round(1800/2), round(2078/2)), 1) #fill the hexagon

masks = [None]*n_file
img = [None]*n_file
gt = [None]*n_file

#Open csv file to store segmentation results
lgd = [ThreshMethod.MEAN.value, ThreshMethod.YEN.value, ThreshMethod.OTSU.value, ThreshMethod.MULTI_OTSU_3.value, ThreshMethod.CUSTOM.value]
radii = [RadiiBkgdRemoval.R_25.value, RadiiBkgdRemoval.R_50.value, RadiiBkgdRemoval.R_250.value, RadiiBkgdRemoval.R_500.value]
headers = ["Baground radii", "Backround removal", "Threshold", "Apply hive mask ", "Erosion", "Dilation", "Filter",
           "accuracy", "precision", "recall", "f1_measure", "dice", "jsi", "Image Name"]
f = open(outpath + path_delimiter + "Results.csv", 'w', newline='')
writer = csv.DictWriter(f, fieldnames=headers)
writer.writeheader()

combi = list(itertools.product([BkgdRemoval.DIVISION, BkgdRemoval.SUBTRACTION], radii, EllipseRadius))
combi = combi + list(itertools.product([BkgdRemoval.NONE], [0], EllipseRadius))
# combi = list(itertools.product([BkgdRemoval.NONE], [0], EllipseRadius))
print(combi)

for b, rad, ell in combi:

    fig_name = f"bkgd{b.name}-r{rad}-ell{ell.value}"
    fig, ax = plt.subplots(nrows=8, ncols=n_file, sharey=True, sharex=True, num=fig_name, figsize=(32, 10))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()

    for i in range(n_file):  #[0]:

        print(f"\nImage {filenames[i]}: {b}-{rad}-{ell}")

        #Open the pictures
        img_full_path = usr_inputs["Path"] + path_delimiter + filenames[i]
        img[i] = np.array(ski.io.imread(img_full_path)) #cropped picture
        gt_full_path = usr_inputs["Path"] + path_delimiter + "ground truth" + path_delimiter + "GT_" + filenames[i]
        gt[i] = np.array(ski.io.imread(gt_full_path))
        gt[i][gt[i] > 0] = 1 
        gray = ski.color.rgb2gray(img[i]) #cropped picture in gray scale
        # plt.figure()
        # plt.imshow(mask)

        #Apply gaussian blur
        r_blur=5*um2pxl #to blur entierly cells
        img_blur = scipy.ndimage.gaussian_filter(gray, r_blur)

        #Compute and Remove background
        if b == BkgdRemoval.NONE:
            img_sub_back = img_blur.copy()
        else:
            img_sub_back = np.zeros(img_blur.shape)
            img_bkgd = scipy.ndimage.gaussian_filter(gray, rad*um2pxl)
            for r in range(img_blur.shape[0]):
                for c in range(img_blur.shape[1]):
                    if b == BkgdRemoval.DIVISION:
                        img_sub_back[r,c] = img_blur[r,c] / img_bkgd[r,c]
                    elif b == BkgdRemoval.SUBTRACTION:
                        img_sub_back[r,c] = img_blur[r,c] - img_bkgd[r,c]
        
        #Rescale denoised images between 0 and 1
        if b != BkgdRemoval.NONE:
            min_val = img_sub_back.flatten().min()
            max_val = img_sub_back.flatten().max()
            #print(j, min_val, max_val)
            img_sub_back = ((img_sub_back - min_val) / (max_val - min_val))
            #print(np.unique(img_sub_back), len(np.unique(img_sub_back)), img_sub_back.flatten().min(), img_sub_back.flatten().max())

                
        #If denoised image is not empty    
        if len(np.unique(img_sub_back)) > 1:

            #Iniate variable
            thresholded = [np.zeros(img_blur.shape) for _ in range(len(lgd))] #will hold thresholded images
            th_value = [None for _ in range(len(lgd))] #will hold threshold values
            accuracy = []; precision = []; recall =[]; f1_measure=[]; dice=[]; jsi=[] #will hold performances

            
            for j,th in enumerate(lgd): 

                #Compute & Apply threshold
                if lgd[j] == ThreshMethod.MEAN.value:
                    th_value[j] = ski.filters.threshold_mean(img_sub_back)
                    thresholded[j] = img_sub_back < th_value[j]
                elif lgd[j] == ThreshMethod.YEN.value:
                    th_value[j] = ski.filters.threshold_yen(img_sub_back)
                    thresholded[j] = img_sub_back < th_value[j]
                elif lgd[j] == ThreshMethod.OTSU.value:
                    th_value[j] = ski.filters.threshold_otsu(img_sub_back)
                    thresholded[j] = img_sub_back < th_value[j]
                elif  lgd[j] == ThreshMethod.MULTI_OTSU_3.value:
                    th_value[j] = ski.filters.threshold_multiotsu(img_sub_back, 3)
                    thresholded[j] = (img_sub_back > th_value[j][0]) & (img_sub_back <= th_value[j][1])
                elif lgd[j] == ThreshMethod.CUSTOM.value:
                    th_value[j], _, _ = custom_threshold(img_sub_back, [], [], "")
                    thresholded[j] = (img_sub_back > th_value[j][0]) & (img_sub_back <= th_value[j][1])

                #Remove everything outside hive
                if params["Apply hive mask "] == True:
                    thresholded[j][hive_mask == 0] = 0
                
                #Erode
                if params["Erosion"] == True:
                    thresholded[j] = ski.morphology.binary_erosion(thresholded[j], footprint=ski.morphology.ellipse(ell.value, ell.value))

                #Dilate
                if params["Dilation"] == True:
                    thresholded[j] = ski.morphology.binary_dilation(thresholded[j], footprint=ski.morphology.ellipse(ell.value, ell.value))

                #Filter
                if params["Filter"]["Status"] == True:
                    labels = ski.measure.label(thresholded[j])
                    regions = ski.measure.regionprops(labels)
                    labs = []
                    for rg in regions:
                        #Find ones to remove
                        if (rg.eccentricity > params["Filter"]["eccentricity"]) or (rg.area_filled <= params["Filter"]["area_min"]) or (rg.area_filled >= params["Filter"]["area_max"]):
                            labs.append(rg.label)
                    #Remove them
                    for k in sorted(labs, reverse=True):
                        #print("Remove:", len(regions), labs, k)
                        del(regions[k-1])
                        thresholded[j][(labels == k)] = 0
                
                    #Fill the regions
                    if  params["Fill Holes"] == True:
                        for rg in regions:
                            r_min, c_min, r_max, c_max = rg.bbox
                            thresholded[j][r_min:r_max, c_min:c_max] = rg.image_filled

                #Compare with ground truth to give a score
                tp = 0 #True positive: segmented as organoid on ground truth and segmentation
                tn = 0 #True negative: not segmented as organoid on ground truth and segmentation
                fp = 0 #False positive: not segmented as organoid on ground truth but on segmentation
                fn = 0 #Pasle negative: segmented as organoid on ground truth but not on segmentation
                for r in range(gt[i].shape[0]):
                    for c in range(gt[i].shape[1]):
                        if gt[i][r,c] == 1 and thresholded[j][r,c] == 1:
                            tp += 1
                        elif gt[i][r,c] == 0 and thresholded[j][r,c] == 0:
                            tn += 1
                        elif gt[i][r,c] == 0 and thresholded[j][r,c] == 1:
                            fp += 1
                        elif gt[i][r,c] == 1 and thresholded[j][r,c] == 0:
                            fn += 1
                #print(f"tp={tp}, tn={tn}, fp={fp}, fn={fn}")
            
                try:
                    accuracy.append((tp + tn) / ( tp + tn + fn + fp)) # % of image that are correctly classified (general, but do not work well on imbalance image, ie when one class is dominant which is the case here)
                except:
                    accuracy.append(np.nan)
                try: 
                    precision.append( tp / (tp + fp) ) # % of organoid that is correctly segmented (focus on overestiation)
                except:
                    precision.append(np.nan)
                try:
                    recall.append( tp / (tp + fn) ) # % of image that is correctly segmented (focus on underestimation)
                except:
                    recall.append(np.nan)
                try:
                    f1_measure.append( 2 * ( precision[j] * recall[j] )/( precision[j] + recall[j] ) ) # focus on level detail and location
                except:
                    f1_measure.append(np.nan)
                try:
                    dice.append( 2*tp/(2*tp + fn + fp) ) # repetability of segementation performance & accuracy of boundary detection & number of correctly segmenented pixel
                except:
                    dice.append(np.nan)
                try:
                    jsi.append( tp / (tp + fp + fn) )#ratio of the overlap between segmentation and ground truth, similar to dice but penalize more incorect detection: dice = (2*jsi)/(1+jsi)
                except:
                    jsi.append(np.nan)
                
                #Display
                ax[j+2, i].imshow(thresholded[j])
                if i == 0:
                    ax[j+2, i].set_ylabel(f"{lgd[j]}", fontsize=15)
        
        else:
            accuracy = [np.nan for _ in range(len(lgd))]
            precision = [np.nan for _ in range(len(lgd))]
            recall = [np.nan for _ in range(len(lgd))]
            f1_measure = [np.nan for _ in range(len(lgd))]
            dice = [np.nan for _ in range(len(lgd))]
            jsi = [np.nan for _ in range(len(lgd))]
               

        ax[0, i].imshow(img[i])
        ax[1, i].imshow(img_sub_back)
        ax[7, i].imshow(gt[i])
        if i == 0:
            ax[0,i].set_ylabel("Original", fontsize=15)
            ax[1,i].set_ylabel("-Bkgd", fontsize=15)
            ax[7,i].set_ylabel("Ground Truth", fontsize=15)
        
        #Write performance in file
        row = {}
        for j in range(len(lgd)):
            row = {"Baground radii": rad,
                   "Backround removal": b.name,
                   "Threshold": lgd[j], 
                   "Apply hive mask ": params["Apply hive mask "], 
                   "Erosion": f"ellipse-{ell.value}-{ell.value}", 
                   "Dilation": f"ellipse-{ell.value}-{ell.value}", 
                   "Filter": True,
                   "accuracy": accuracy[j],
                   "precision": precision[j],
                   "recall": recall[j],
                   "f1_measure": f1_measure[j],
                   "dice": dice[j],
                   "jsi": jsi[j],
                   "Image Name": filenames[i]}
            if b == BkgdRemoval.NONE:
                row.update({"Baground radii": np.nan})
            writer.writerow(row)

    fig.savefig(outpath + path_delimiter + fig_name)
    plt.close(fig) 

f.close()
quit()


    # #Interesting ones when not removing background: Yen, mean, otsu
    # #Set H2M150-short_D6: 
    # lgd = ["Original", "Otsu", "Mean", "Minimum", "Niblack", "Sauvola", "Yen"]
    # th = [np.zeros(img_blur.shape) for _ in range(len(lgd))]
    # th_value = []
    # th[0] = img_blur
    # th_value.append(ski.filters.threshold_otsu(img_blur))
    # th_value.append(ski.filters.threshold_mean(img_blur))
    # th_value.append(ski.filters.threshold_minimum(img_blur))
    # th_value.append(ski.filters.threshold_niblack(img_blur))
    # th_value.append(ski.filters.threshold_sauvola(img_blur))
    # th_value.append(ski.filters.threshold_yen(img_blur))
    # for j in range(len(th_value)):
    #     th[j+1] = img_blur > th_value[j]
    # fig1 = plt.figure()
    # ax1=[]
    # for j in range(len(th)):
    #     if j < len(th_value):
    #         print(type(th_value[j]))
    #         print(isinstance(th_value[j], np.ndarray))
    #         if isinstance(th_value[j], np.ndarray) == False:
    #             print(th_value[j])
    #         th[j+1] = img_blur > th_value[j]
    #     ax1.append(fig1.add_subplot(2,4,j+1))
    #     ax1[-1].set_title(lgd[j])
    #     plt.imshow(th[j])

    #Intersting ones when removing background: minimum and sauvolar for r=100 to r=500
    #For set H2M15-short_D6: 
    
#     #Background
#     img_bkgd = [scipy.ndimage.gaussian_filter(mask, 25*um2pxl), 
#                 scipy.ndimage.gaussian_filter(mask, 100*um2pxl), 
#                 scipy.ndimage.gaussian_filter(mask, 300*um2pxl),
#                 scipy.ndimage.gaussian_filter(mask, 500*um2pxl)]
#     r_blur=25*um2pxl #to blur the mesh and background in general
#     img_bckgd_s = scipy.ndimage.gaussian_filter(mask, r_blur)
#     r_blur=100*um2pxl
#     img_bckgd_h = scipy.ndimage.gaussian_filter(mask, r_blur)
#     r_blur=300*um2pxl
#     img_bckgd_H = scipy.ndimage.gaussian_filter(mask, r_blur)
#     for j in range(len(img_bkgd)):
#         img_bkgd[j][mask == 0] = 0
#         # f = plt.figure()
#         # plt.imshow(img_bkgd[j])

#     #Remove background
#     img_sub_back = [np.zeros(img_blur.shape, dtype='float64'),
#                     np.zeros(img_blur.shape, dtype='float64'),
#                     np.zeros(img_blur.shape, dtype='float64'),
#                     np.zeros(img_blur.shape, dtype='float64'),
#                     np.zeros(img_blur.shape, dtype='float64')]
#     img_sub_back[0] = img_blur.copy() #0 element for no bkgd removal
#     for r in range(img_blur.shape[0]):
#         for c in range(img_blur.shape[1]):
#             diff_s = img_blur[r,c] - img_bkgd[0][r,c]*0.9
#             diff_h = img_blur[r,c] - img_bkgd[1][r,c]*0.9
#             diff_H = img_blur[r,c] - img_bkgd[2][r,c]*0.9
#             diff_HH = img_blur[r,c] - img_bkgd[3][r,c]*0.9
#             if diff_s < 0:
#                 img_sub_back[1][r,c] = 0
#             else:
#                 img_sub_back[1][r,c] = diff_s
#             if diff_h < 0:
#                 img_sub_back[2][r,c] = 0
#             else:
#                 img_sub_back[2][r,c] = diff_h
#             if diff_H < 0:
#                 img_sub_back[3][r,c] = 0
#             else:
#                 img_sub_back[3][r,c] = diff_H
#             if diff_HH < 0:
#                 img_sub_back[4][r,c] = 0
#             else:
#                 img_sub_back[4][r,c] = diff_HH
#     # for j in range(len(img_sub_back)):
#     #     f = plt.figure()
#     #     plt.imshow(img_sub_back[j])
#     c = 8
#     r = 5
#     #th = [ [np.zeros(img_blur.shape)]*r ] * c
#     th = [[np.zeros(img_blur.shape) for _ in range(r)] for _ in range(c)]
#     print(len(th), len(th[0]), th[0][0].shape)
#     for n in range(r):
#         th[0][n] = img_blur
#         th[1][n] = img_sub_back[n]

#         try:
#             otsu = ski.filters.threshold_otsu(img_sub_back[n])
#         except:
#             otsu = 1
#         print("Otsu=", otsu)
#         th[2][n] = img_sub_back[n] > otsu

#         try:
#             meann = ski.filters.threshold_mean(img_sub_back[n]) 
#         except:
#             meann = 1
#         print("Mean=", meann)
#         th[3][n] = img_sub_back[n] > meann

#         try:
#             minim =ski.filters.threshold_minimum(img_sub_back[n])
#         except:
#             minim = 1
#         print("Minimum=", minim)
#         th[4][n] = img_sub_back[n] > minim

#         try:
#             niblack = ski.filters.threshold_niblack(img_sub_back[n])
#         except:
#             niblack = 1
#         th[5][n] = img_sub_back[n] > niblack

#         try:
#             sauvola = ski.filters.threshold_sauvola(img_sub_back[n])
#         except:
#             sauvola = 1
#         th[6][n] = img_sub_back[n] > sauvola

#         try:
#             yen = ski.filters.threshold_yen(img_sub_back[n])
#         except:
#             yen = 1
#         print("Yen=", yen)
#         th[7][n] = img_sub_back[n] > yen

#     lgd = ["Original", "No bkgd", "Otsu", "Mean", "Minimum", "Niblack", "Sauvola", "Yen"]
#     met = ["no r", "r=25", "r=100", "r=300", "r=500"]

#     fig2 = plt.figure(filenames[i], figsize=(10,10))
#     ax2 = []
#     j=0
#     for y in range(r):
#         for x in range(c):
#             ax2.append(fig2.add_subplot(r,c,j+1))
#             ax2[-1].set_title(lgd[x]+met[y])
#             plt.imshow(th[x][y])
#             j=j+1
#     plt.savefig(usr_inputs["Path"]+path_delimiter+"SegmTest"+filenames[i])

#     # Find the different regions, label it
#     label = [[np.zeros(img_blur.shape) for _ in range(r)] for _ in range(c)]
#     regions = [[None for _ in range(r)] for _ in range(c)]
#     fig3 = plt.figure(filenames[i], figsize=(10,10))
#     ax3 = []
#     j=0
#     for n in range(r):
#         for m in range(c):
#             if m == 0: label[m][n] = img_blur
#             elif m == 1: label[m][n] = img_sub_back[n]
#             else:
#                 label[m][n] = ski.measure.label(th[m][n])
#                 regions[m][n] = ski.measure.regionprops(label[m][n])
#             ax3.append(fig3.add_subplot(r,c,j+1))
#             ax3[-1].set_title(lgd[m]+met[n])
#             plt.imshow(label[m][n])
#             j=j+1
#     plt.savefig(usr_inputs["Path"]+path_delimiter+"LabelTest"+filenames[i])

#     # Keep only the objects that could be organoids:
#     # circularity in between 0.3-1
#     # area in between 0.007-1.5 mm²
#     fig4 = plt.figure(filenames[i], figsize=(10,10))
#     ax4 = []
#     j=0
#     for n in range(r):
#         for m in range(c):
#             if m not in range(2) and len(regions[m][n]) > 0: 
#                 labs = []
#                 for rg in regions[m][n]:
#                     print(rg.eccentricity, rg.area_filled)
#                     if (rg.eccentricity >= 0.8) or (rg.area_filled <= 50000) or (rg.area_filled >= 5000000):
#                         labs.append(rg.label)
#                 for k in sorted(labs, reverse=True):
#                     del(regions[m][n][k-1])
#                     label[m][n][(label[m][n] == k)] = 0
#                 print(regions)
#             ax4.append(fig4.add_subplot(r,c,j+1))
#             ax4[-1].set_title(lgd[m]+met[n])
#             plt.imshow(label[m][n])
#             j = j + 1
#     plt.savefig(usr_inputs["Path"]+path_delimiter+"LabelRmTest"+filenames[i])
            

# plt.show()
        


    #Between Yen and Otsu
    #Yen= for uneven illumination, when Otsu struggles
    #Otsu=
    #Mean=
    #fig, ax = ski.filters.try_all_threshold(img_blur, figsize = (10, 6), verbose=False)

