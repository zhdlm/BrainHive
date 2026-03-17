import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
from os import listdir
from PIL import Image
from joblib import Parallel, parallel_backend, delayed

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
# Angle= 0

try:
    usr_inputs = open("inputs.txt").read()
    txt = usr_inputs.split('\n')
    path = txt[0].split('= ')[1]
    filename_pattern = txt[1].split('= ')[1]
    microscope = txt[2].split('= ')[1]
    magnification = txt[3].split('= ')[1]
    snake = txt[4].split('= ')[1]
    gray = txt[5].split('= ')[1]
    angle = float(txt[6].split('= ')[1])
    aquisition_order = txt[7].split('= ')[1]
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
print(filenames)
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
    # print('Drawing hexagon outlines: image size=', image_shape, 'center=', 
    #       center, 'side=', s, 'thickness=', thickness)

    height = image_shape[0]
    width = image_shape[1]
    img = np.zeros((height, width), dtype=np.int8)

    cx = center[1]
    cy = center[0]
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    x_vertices = cx + s * np.cos(angles)
    y_vertices = cy + s * np.sin(angles)

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

def rotate_image(img, angle):
    return ski.transform.rotate(img, angle, resize=True, preserve_range=True)


#############################################################################################
# Crop picture around hive
#############################################################################################

#Rescale ratio
scale_ratio=6

#Open pictures, resize it, put in grayscale and store it in a list
img_rs = [None]*len(filenames) ; shp= [None]*len(filenames)
img = [None]*len(filenames)
for i in range(len(filenames)):
    img_full_path = path + '\\' + filenames[i]
    img[i] = np.array(ski.io.imread(img_full_path))
    img_sz = img[i].shape
    img_gray = ski.color.rgb2gray(img[i])
    img_rs[i] = ski.transform.resize(img_gray, (round(img_sz[0]/scale_ratio), round(img_sz[1]/scale_ratio)))
    shp[i]=img_rs[i].shape
    # img_rs[i].setflags(write=True)
    # img[i].setflags(write=True)
    print('Opening image, adujsting to grayscale and rescaling to', scale_ratio, ': ', i+1, '/', len(filenames), end='\r')
print()
img_type = img[0].dtype #store initial pixel value type
del(img_gray)

#Verify the shapes are consistent
if len(set(shp)) > 1:
    print("No pictures found.\nScript aborted.")
    quit()
else:
    img_rs_sz = img_rs[0].shape
    print('Pictures successfully oppened.')
    print('Original image dimension: ', img_sz)
    print('Image dimension after ', scale_ratio,' times rescale: ', img_rs_sz)

#Parameters depending on user inputs. Need to automate this of read doc to get good values
um2pxl = 0.9 # 1 µm corresponds to 0.5645 pxl
pxl2um = 1.1 # 1 pxl corresponds to 1.7714 µm
h = 2000 #height of hive in mm
s = round(h*um2pxl/(2*np.sin(np.pi/3)))
d = 2*s
thickness = 5

#Creation of a hive image with good dimensionfor skimage functions
h_conv = round(h*um2pxl/scale_ratio) #0.67
s_conv = round(h_conv/(2*np.sin(np.pi/3))) #side of hive
d_conv = 2*s_conv #long diagonal of the hive
print('Hive dimensions in pixel rescaled for microscope ', microscope, 'at magnification ', magnification, 
      ': height=', h_conv, ', side=', s_conv, ', diagonal=', d_conv)
hive_rs = hexagon_outline_ndarray(img_rs_sz, [img_rs_sz[0]/2, img_rs_sz[1]/2], s_conv, thickness)
# plt.imshow(hive_img_conv)
# plt.colorbar()
# plt.show()
# quit()

#Determine good angle
angle_r=5
angle_step=0.5
best_angles = np.empty(len(img_rs))
for i in range(len(img_rs)):
#for i in idx: 
    print('Determining angle to have picture in good orientation:', i+1, '/', len(img_rs), end='\r')
    conv = [None]*len(np.arange(-angle_r+angle, angle_r+angle, angle_step))
    v_max = [None]*len(np.arange(-angle_r+angle, angle_r+angle, angle_step))
    #Convolution at different angles
    j=0
    for a in np.arange(-angle_r+angle, angle_r+angle, angle_step):
        conv[j] = scipy.signal.fftconvolve(img_rs[i], ski.transform.rotate(hive_rs, a), mode='same') #convolution in fourrier space for delay
        v_max[j] = conv[j].max().max() #
        # plt.figure('Angle ' + str(a))
        # plt.imshow(conv[j])
        j+=1

    #Find best angle
    i_best = v_max.index(max(v_max))
    a_best = np.arange(-angle_r+angle, angle_r+angle, angle_step)[i_best]
    best_angles[i] = a_best
print()

#Find best angle for rotation
a_best = np.nanmedian(best_angles)
i_best = np.where(np.arange(-angle_r+angle, angle_r+angle, angle_step) == a_best)
print('Best angle is:', a_best)

#Parallelization of rotation of reisized and original pictures --> not succeeding to make it work
[print(i.flags.writeable) for i in img]
[print(i.flags.writeable) for i in img_rs]
# print(parallel_backend().__enter__())  # Just to print backend info
img = [np.array(i, copy=True) for i in img]
img_rs = [np.array(j, copy=True) for j in img_rs]
with parallel_backend('loky'):
    img_rot = Parallel(n_jobs=2)(
        delayed(rotate_image)(i, 90) for i in img
    )
with parallel_backend('loky'):
    img_rs_rot = Parallel(n_jobs=2)(
        delayed(rotate_image)(i, 90) for i in img_rs
    )
# img_rot = Parallel(n_jobs=-1)(delayed(rotate_image)(i, -a_best) for i in img)
# img_rs_rot = Parallel(n_jobs=-1)(delayed(rotate_image)(j, -a_best) for j in img_rs)
img = img_rot
im_rs = img_rs_rot
del(img_rot, img_rs_rot)
quit()

# #Rotation of resized and original pictures == 12.029sec
# for i in range(len(img_rs)):
#     img_rs[i] = ski.transform.rotate(img_rs[i], -a_best, preserve_range=True, resize=True)
#     img[i] = ski.transform.rotate(img[i], -a_best, preserve_range=True, resize=True)
#     img[i].astype(img_type)
#     #Adjust dimension after rotation to have at least the dimension (y,x) as: (hexagon height, hexagon diagonal)
#     if img[i].shape[0] - h*um2pxl < 0:
#         # print('height of picture is to short to fit hexagon height')
#         w = h*um2pxl - img[i].shape[0] +0.1 #addition of 0.1 to be sure 
#         tmp = np.empty((round(h*um2pxl), img[i].shape[1], 3), dtype=img_type)
#         tmp[round(w/2):round(img[i].shape[0]+w/2), :, :] = img[i]
#         img[i] = tmp
#     if img[i].shape[1] - d < 0:
#         # print('width of picture is to short to fit hexagon diagonal')
#         w = d - img[i].shape[1] +0.1
#         tmp = np.empty((img[i].shape[0], round(d), 3), dtype=img_type)
#         tmp[:, round(w/2):round(img[i].shape[1]+w/2), :] = img[i]
#         img[i] = tmp
#     # plt.figure('rotation '+filenames[i])
#     # plt.imshow(img[i])
#     # print(img[i].shape)
# plt.show()


#Crop
#for i in idx:
for i in range(len(img_rs)):
    print('Computing hexagon barycenter and cropping picture:', i+1, '/', len(img_rs), end='\r')

    #Pre-process of the picture
    blur = scipy.ndimage.gaussian_filter(img_rs[i], 5/scale_ratio) #blur pictire to smooth edges
    otsu_thresh = ski.filters.threshold_otsu(blur) #compute the otsu automatic threshold
    thresh = np.empty(blur.shape) 
    thresh[blur >= otsu_thresh] = 0 #value below thresh =0
    thresh[blur < otsu_thresh] = 1 #value above thresh =1
    conv = scipy.signal.fftconvolve(thresh, hive_rs, mode='same') #perform convolution of hive mask and pre-processes imaged (blurred, thesrloded and rotated)
    # plt.figure('conv'+filenames[i])
    # plt.imshow(conv)
    tmp = conv
    v_max = tmp.max().max() #compute maximum pixel value on the results of convolution
    tmp[tmp < v_max] = 0
    # plt.figure('conv cut'+filenames[i])
    # plt.imshow(tmp)
    del(blur, thresh)

    #Compute barycenter of the convolution image
    if sum(tmp.sum(axis=1)) != 0:
        y_c = sum(tmp.sum(axis=1)*range(tmp.shape[0]))/sum(tmp.sum(axis=1)) #projection on y axis and computing barycenter of y
        x_c = sum(tmp.sum(axis=0)*range(tmp.shape[1]))/sum(tmp.sum(axis=0)) #projection on x axis and computing barycenter of x
    else: #in case of empty picture
        y_c = 0
        x_c = 0

    #Convert the barycenter at good scale
    y_c = round(img[i].shape[0]*y_c/conv.shape[0])
    x_c = round(img[i].shape[1]*x_c/conv.shape[1])
    # print('Coordinate of barycenter (x,y): (', y_c, ',', x_c, ')')
    del(conv, tmp)

    #Create hive mask for the crop with center as barycenter of convolution picture
    hive_mask =  hexagon_outline_ndarray(img[i].shape, [y_c, x_c], s, thickness=thickness) #recreate the hexagon outlines at good dimension
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
    #y values
    if round(y_c - h*um2pxl/2) < 0:
        y_cut1 = 0
        y_cut2 = round(h*um2pxl)
        # print('negative y')
    elif round(y_c + h*um2pxl/2) > img[i].shape[0]:
        y_cut1 = round(img[i].shape[0] - h*um2pxl)
        y_cut2 = img[i].shape[0]
        # print('y larger than picture')
    else:
        y_cut1 = round(y_c - h*um2pxl/2)
        y_cut2 = round(y_c + h*um2pxl/2)
        # print('y ok')
    #x values
    if round(x_c - d/2) < 0:
        x_cut1 = 0
        x_cut2 = d
        # print('negative x')
    elif round(x_c + d/2) > img[i].shape[1]:
        x_cut1 = round(img[i].shape[1] - d)
        x_cut2 = img[i].shape[1]
        # print('x larger than picture')
    else:
        x_cut1 = round(x_c - d/2)
        x_cut2 = round(x_c + d/2)
        # print('x ok')
    img[i][hive == 0] = 0 #assign all values outside hive to 0
    img[i] = img[i][y_cut1:y_cut2, x_cut1:x_cut2, :] #crop around h and d
    # plt.figure('Image cropped'+filenames[i])
    # plt.imshow(img[i])
plt.show()
print('\nCrop Succes')

#############################################################################################
# Montage
#############################################################################################

# Substrate is definied as follow:
# 1st column: 3 hives
# 2nd column: 4 hives
# 3rd column: 5 hives
# 4th column: 3 hives
# 5th column: 4 hives

#Image acquisition order:
if aquisition_order == 'bottom':
    print('bottom')
    idx = [[11,3,2], [12,10,4,1], [18,13,9,5,0], [17,14,8,6], [16,15,7]]
elif aquisition_order == 'bottom right':
    print('bottom right')
    idx = [[2,1,0], [3,4,5,6], [11,10,9,8,7], [12,13,14,15], [18,17,16]]
idx_flat = [item for sublist in idx for item in sublist]
print('image qquisition order: ', idx_flat)

#Reconstruct final image
img_sz = img[0].shape
print('Cropped Images dimension (y,x):', img_sz)
montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=img_type)
print('Montage dimension (y,x):', montage.shape)
print(type(img[0][0,0,0]))
i=0 #counting pictures in following nested loops
r=0 #row on the substrate
for c in [0,1,2,1,0]:
    #Initiate location of the first picture of the row
    y_min = round((5*img_sz[0] - (c+3)*img_sz[0])/2)
    y_max = round(y_min + img_sz[0])
    x_min = img_sz[1]*(r)
    x_max = x_min + img_sz[1]

    for elt in range(len(idx[r])):
        print('c=', c, 'elt=', elt, 'coord= ', y_min, y_max, x_min, x_max)
        montage[y_min:y_max, x_min:x_max, :] = img[idx_flat[i]] #add picture to montage at good location
        #Set paramaters for next picture from the row
        i += 1
        y_min += img_sz[0]
        y_max = y_min + img_sz[0]
        # plt.figure('Montage')
        # plt.imshow(montage)
        # plt.show()
    r+=1
plt.figure('Montage')
plt.imshow(montage)
plt.show()
try:
    out_path = path +'\\Montage_' + filename_pattern + '_' + str(a_best) + 'degree.jpeg'
    ski.io.imsave(out_path, montage)
    print('Imaged saved as: ', out_path)
except:
    print('Image could not be saved in ', out_path)