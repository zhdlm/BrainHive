import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
from os import listdir
from PIL import Image
import re
import statistics

#############################################################################################
# Read user inputs & search pictures
#############################################################################################

# Read user inputs written as:
# Path= <pathname>
# Filenames= <filenames>
# Microscope= Leica
# Magnification= 5x
# Snake= True
# Color output= True

try:
    usr_inputs = open("inputs.txt").read()
    txt = usr_inputs.split('\n')
    path= txt[0].split('= ')[1]
    filename_pattern= txt[1].split('= ')[1]
    microscope= txt[2].split('= ')[1]
    magnification= txt[3].split('= ')[1]
    snake= txt[4].split('= ')[1]
    gray= txt[5].split('= ')[1]
    del(usr_inputs, txt)
    print("\nUser inputs loaded")
except:
    print("\nError in user inputs.\nScript aborted.")
    quit()

#Search all pictures
files = listdir(path)
filenames=[]
valid_extension=['tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
for f in files:
    if (any(ext in f for ext in valid_extension)) and (filename_pattern in f): filenames.append(f)
#Verify there is the good number of picture and sort them by alphabetical order
if len(filenames) > 0:
    filenames.sort()
    if len(filenames) == 19:
        print("Expected number of pictures found")
    elif len(filenames) > 19:
        print("More than 19 pictures are found.\nScript aborted.")
        quit()
    else:
        print("Less than 19 pictures are found, absent pictures will be replaced by empty picture in Montage.")
else:
    print("No pictures found.\nScript aborted.")
    quit()


def hexagon_outline_ndarray(image_shape, center, s, thickness=1):
    """
    Draws a hexagon outline (not filled) into a 2D ndarray.
    
    Parameters:
        image_shape (list): [height, width] of the output ndarray.
        center (list): [y, x] center of the hexagon.
        s (float): side length of the hexagon.

    Returns:
        np.ndarray: ndarray with hexagon outline written as 1s.
    """
    print('Drawing hexagon outlines: image size=', image_shape, 'center=', 
          center, 'side=', s, 'thickness=', thickness)

    height = image_shape[0]
    width = image_shape[1]
    img = np.zeros((height, width), dtype=np.uint8)

    cx = center[1]
    cy = center[0]
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    x_vertices = cx + s * np.cos(angles)
    y_vertices = cy + s * np.sin(angles)
    # #Force the values to be inside image
    # x_vertices[x_vertices < 0] = 0.1
    # y_vertices[y_vertices < 0] = 0.1
    # x_vertices[x_vertices > width] = width
    # y_vertices[y_vertices > height] = height
    # print(angles, x_vertices, y_vertices)

    # Draw lines between consecutive vertices
    for i in range(6):
        x0, y0 = int(round(x_vertices[i])), int(round(y_vertices[i]))
        x1, y1 = int(round(x_vertices[(i + 1) % 6])), int(round(y_vertices[(i + 1) % 6]))
        rr, cc = ski.draw.line(y0, x0, y1, x1)  # Notice (row, col) = (y, x)
        # For each pixel in the line, draw a small disk of radius thickness//2
        for r, c in zip(rr, cc):
            dr, dc = ski.draw.disk((r, c), radius=thickness // 2, shape=img.shape)
            img[dr, dc] = 1

    return img


#############################################################################################
# Crop picture around hive
# Bad crop for: 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18
# Bad crop after addition of post-process and selection of median of best angles for conv & rotation: 4, 8, 9, 12, 13, 14, 15  
# Bad crop after addition of post-process and selection of median of best angles for rotation only: 4, 8, 9, 11, 12, 13, 14, 15
#############################################################################################

#Rescale ratio
scale_ratio=4

#Open pictures, resize it, put in grayscale and store it in a list
img_rs = [None]*len(filenames) ; shp= [None]*len(filenames)
img = [None]*len(filenames)
for i in range(len(filenames)):
    img_full_path = path + '\\' + filenames[i]
    img[i] = np.array(ski.color.rgb2gray(ski.io.imread(img_full_path)))
    img_sz = img[i].shape
    img_rs[i] = ski.transform.resize(img[i], (round(img_sz[0]/scale_ratio), round(img_sz[1]/scale_ratio)))
    shp[i]=img_rs[i].shape
    #print('\nImage ', filenames[i], ':\nSize= ', img_rs[i].shape)

#Verify the shapes are consistent
if len(set(shp)) > 1:
    print("No pictures found.\nScript aborted.")
    quit()
else:
    img_sz = img_rs[0].shape
    print('Pictures successfully oppened.')
    print('Original image dimension: ', img[i].shape)
    print('Image dimension after ', scale_ratio,' times rescale: ', img_sz)

#Parameters depending on user inputs
um2pxl = 0.9 # 1 µm corresponds to 0.5645 pxl
pxl2um = 1.1 # 1 pxl corresponds to 1.7714 µm
h = 2000 #height of hive in mm
s = round(h*um2pxl/(2*np.sin(np.pi/3)))
d = 2*s
thickness = 5

#Creation of a hive image with good dimension
h_conv = round(h*um2pxl/scale_ratio) #0.67
s_conv = round(h_conv/(2*np.sin(np.pi/3))) #side of hive
d_conv = 2*s_conv #long diagonal of the hive
print('Hive dimensions in pixel rescaled for microscope ', microscope, 'at magnification ', magnification, 
      ': height=', h_conv, ', side=', s_conv, ', diagonal=', d_conv)
hive_img = hexagon_outline_ndarray(img_sz, [img_sz[0]/2, img_sz[1]/2], s_conv, thickness)
# plt.imshow(hive_img)
# plt.colorbar()
# plt.show()

#Determine good angle
idx = [0, 1, 15]
angle_r=5
angle_step=0.5
angles = np.empty(len(img_rs))
#for i in range(len(img_rs)):
for i in idx: 
    print('\nConvolution of hive and rescaled image ', filenames[i])
    conv = [None]*len(np.arange(-angle_r, angle_r, angle_step))
    v_max = [None]*len(np.arange(-angle_r, angle_r, angle_step))
    #Convolution at different angles
    j=0
    for a in np.arange(-angle_r, angle_r, angle_step):
        conv[j] = scipy.signal.fftconvolve(img_rs[i], ski.transform.rotate(hive_img, a), mode='same') #convolution in fourrier space for delay
        v_max[j] = conv[j].max().max() #
        # plt.figure('Angle ' + str(a))
        # plt.imshow(conv[j])
        j+=1

    #Find best angle
    i_best = v_max.index(max(v_max))
    a_best = np.arange(-angle_r, angle_r, angle_step)[i_best]
    angles[i] = a_best
    # plt.figure('Hive with best angle for '+ filenames[i])
    # plt.imshow(ski.transform.rotate(hive_img, a_best))

#Find best angle for rotation + convolution
a_best = np.nanmedian(angles)
i_best = np.where(np.arange(-angle_r, angle_r, angle_step) == a_best)

#Crop
for i in idx:
# for i in range(len(img_rs)):

    # Post process on convolution: keep higher pixel values
    conv = scipy.signal.fftconvolve(img_rs[i], ski.transform.rotate(hive_img, angles[i]), mode='same')
    plt.figure('convolution pre process'+filenames[i])
    plt.imshow(conv)
    tmp = conv
    # tmp[tmp< np.percentile(tmp, 92)] = 0 #keep only the 94 percentile, ie highest values, to remove noise
    # tmp[tmp > 0] = tmp.max().max() #set all high values at max intensity
    label_img = ski.measure.label(tmp) #find the different regions in the convolution image
    regions = ski.measure.regionprops(label_img) #extract information on those regions
    areas = [j.area_filled for j in regions] #areas of the different regions
    idx_area_max = areas.index(max(areas))
    conv[label_img != idx_area_max+1] = 0 #keep only the biggest region in the convolution image
    plt.figure('Region '+filenames[i])
    plt.imshow(label_img)
    plt.figure('convolution'+filenames[i])
    plt.imshow(conv)
    plt.colorbar()

    # Post process search for hexagon in convolution picture
    # conv = scipy.signal.fftconvolve(img_rs[i], ski.transform.rotate(hive_img, angles[i]), mode='same')
    # tmp = conv
    # tmp[tmp< np.percentile(tmp, 92)] = 0 #keep only the 94 percentile, ie highest values, to remove noise
    # tmp[tmp > 0] = tmp.max().max() #set all high values at max intensity
    # plt.figure('conv cut' + filenames[i])
    # plt.imshow(conv)
    # hive_mask_conv =  hexagon_outline_ndarray(conv.shape, [conv.shape[0]/2, conv.shape[1]/2], s/(scale_ratio*4), thickness=thickness)
    # plt.figure('hive mask conv' + filenames[i])
    # plt.imshow(hive_mask_conv)
    # conv_conv = scipy.signal.fftconvolve(tmp, hive_mask_conv, mode='same')
    # plt.figure('conv of conv' + filenames[i])
    # plt.imshow(conv_conv)
    # areas = [j.area_filled for j in regions] #areas of the different regions
    # idx_area_max = areas.index(max(areas))
    # #conv[label_img != idx_area_max+1] = 0 #keep only the biggest region in the convolution image
    # plt.figure('Region '+filenames[i])
    # plt.imshow(label_img)
    # plt.figure('convolution'+filenames[i])
    # plt.imshow(conv)
    # plt.colorbar()

    # #Compute barycenter of the convolution image
    # y_c = sum(conv.sum(axis=1)*range(conv.shape[0]))/sum(conv.sum(axis=1)) #projection on y axis and computing barycenter of y
    # x_c = sum(conv.sum(axis=0)*range(conv.shape[1]))/sum(conv.sum(axis=0)) #projection on x axis and computing barycenter of x
    # print('Best angle: ', a_best, '\nCoordinate of barycenter (x,y): (', y_c, ',', x_c, ')')

    # #Convert the barycenter at good scale
    # y_c = round(img[i].shape[0]*y_c/conv.shape[0])
    # x_c = round(img[i].shape[1]*x_c/conv.shape[1])
    # del(conv, tmp, label_img, regions)

    # #Create hive mask for the crop with center as barycenter of convolution picture
    # hive_mask =  hexagon_outline_ndarray(img[i].shape, [y_c, x_c], s, thickness=thickness) #recreate the hexagon outlines at good dimension
    # hive = ski.segmentation.flood_fill(hive_mask, (y_c, x_c), 1) #fill the hexagon
    # print('Force int values. List of possible values: ', np.unique(hive))   
    # hive = hive.astype(int)
    # #Force inner part to be set to at 1
    # if hive[y_c, x_c] == 0:
    #     hive = np.invert(hive)
    # # plt.imshow(hive)
    # # plt.colorbar()
    # # plt.show()

    # #Compute x and y values where crop should occure:
    # #y values
    # if round(y_c - h*um2pxl/2) < 0:
    #     y_cut1 = 0
    #     y_cut2 = round(h*um2pxl)
    # elif round(y_c + h*um2pxl/2) > img[i].shape[0]:
    #     y_cut1 = round(img[i].shape[0] - h*um2pxl)
    #     y_cut2 = img[i].shape[0]
    # else:
    #     y_cut1 = round(y_c - h*um2pxl/2)
    #     y_cut2 = round(y_c + h*um2pxl/2)
    # #x values
    # if round(x_c - d/2) < 0:
    #     x_cut1 = 0
    #     x_cut2 = d
    # elif round(x_c + d/2) > img[i].shape[1]:
    #     x_cut1 = round(img[i].shape[1] - d)
    #     x_cut2 = img[i].shape[1]
    # else:
    #     x_cut1 = round(x_c - d/2)
    #     x_cut2 = round(x_c + d/2)
    # print(x_cut1, x_cut2, y_cut1, y_cut2)
    # img[i] = ski.transform.rotate(img[i], -a_best) #rotate original image to have it in good orientation
    # img[i][hive == 0] = 0 #assign all values outside hive to 0
    # img[i] = img[i][y_cut1:y_cut2, x_cut1:x_cut2] #crop around h and d
    # # plt.figure('Image cropped'+filenames[i])
    # # plt.imshow(img[i])
plt.show()

# plt.imshow(conv[0])
# plt.colorbar()
# plt.show()
