import os
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from scipy.ndimage import gaussian_filter, median_filter
from skimage.transform import rescale
from skimage.filters import threshold_otsu, threshold_yen, rank, threshold_local, threshold_multiotsu, sato
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_erosion, remove_small_objects, binary_closing, binary_opening, skeletonize
from skimage.morphology import disk, ellipse
from skimage.measure import label
from skimage.draw import line_nd
# import kimimaro
import czifile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd
from pprint import pprint
import json
import skan
import networkx
import napari

VERSION = "V1"
VESSEL_DIAM = 10 #in µm
print(f"=================================")
print(f"      MICRO-VESSEL ANALYSIS      ")
print(f"=================================")

#0. User Inputs
day_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal"
out_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal-Analysis"
filename = "Exp004-E_F5_Seqlong_x10_GFP-BF-Cy3-DAPI_16bit-1024-n4-s7_D14_2025_12_01__16_48_47.czi"
fluo_endo = "GFP"
fluo_chans = ["GFP","CY3","BF","DAPI"]
z_start = 3
z_end = -1 #set to -1 if you want to go until the end of the stack
print(f"User inputs stored.")

#############################
# 1. LOADING
#############################

#1.a Create CZI obejct and extract relevant info from metadata
czi = czifile.CziFile(os.path.join(day_path, filename)) # CZI file + metadata
metadata = czi.metadata(raw=False)
pxls2um = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
xy_pxl2um = pxls2um[0]["Value"]*1000000
z_pxl2um = pxls2um[2]["Value"]*1000000
z_step_um = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Z"]["Positions"]["Interval"]["Increment"]
print(f"X-Ypxl={xy_pxl2um}um; Zpxl={z_pxl2um}um")


#1.b Load images
full_stack = czi.asarray(os.path.join(day_path, filename)) #Full stack of shape (?, ?, ?, Channel, Z, Y, X, RGB)
dims = (full_stack.shape[5], full_stack.shape[6])
endo_stack = full_stack[0,0,0, fluo_chans.index(fluo_endo), z_start:z_end, :, :, 0]
depth = endo_stack.shape[0]
print(f"Stack loaded (Z,Y,X)=({depth}, {dims[0]}, {dims[1]})")

#1.c Metrics extraction: depth, I_mean_raw(z), I_std_raw(z), I_med_raw(z), I_mad_raw(z)

#############################
# 2. PRE-PROCESSING
#############################

#2.a Remove noise
# r_blur_xy = (VESSEL_DIAM/10)/xy_pxl2um
# r_blur_z = (VESSEL_DIAM/10)/z_pxl2um
# r_blur = (r_blur_xy, r_blur_xy, r_blur_z)
# endo_stack_denoised = median_filter(endo_stack, size=50)
# print(f"Denoised using gaussian blur sigmas=({r_blur_z},{r_blur_xy},{r_blur_xy}) (Z,Y,X)")

#2.b Improve contrast (usefull for uneven illumination)
endo_stack_contrast = equalize_adapthist(endo_stack)
print(f"Improve local contrast using CLAHE (kernel_size=1/8width, clip_lim=0.01, nbins=256)")

#2.c Vessel Enhancement (sato seems better than frangi)
vessel_diam = [2, 4, 6, 8, 10]
vessel_diam_pxl = [round(xy_pxl2um*elt/2, 1) for elt in vessel_diam]
anisotropy = z_pxl2um / xy_pxl2um
#Anistotripic voxel -> slice by slice
if anisotropy > 4:
    print("High anisotropic voxel -> Sato performed slice by slice")
    endo_vessel_sato = np.ndarray((depth, dims[0], dims[1]))
    for z in range(depth):
        endo_vessel_sato[z, :, :] = sato(endo_stack_contrast[z, :, :], sigmas=range(1,10), black_ridges=False)
#Midly anistotropic -> reslice for isotropic voxel
elif anisotropy >= 2:
    print("Moderate anisotropic voxel -> Reslice and Sato performed of resliced stack")
    print("not implemented yet")
    endo_vessel_iso = rescale(endo_stack_contrast, (z_pxl2um/xy_pxl2um, 1, 1), preserve_range=True)
    endo_vessel_sato = sato(endo_vessel_iso, sigmas=range(1,10), black_ridges=False)
    pass
#Isotropic voxel
else:
    print("Low anisotropic voxel -> Sato performed on stack")
    endo_vessel_sato = sato(endo_stack_contrast, sigmas=range(1,10), black_ridges=False)

#############################
# 3. THRESHOLD
#############################

otsu_thresh = threshold_otsu(endo_vessel_sato)
endo_vessel_thresh = endo_vessel_sato > otsu_thresh
print(f"Otsu Thesrhold on stack: otsu={otsu_thresh}")

#############################
# 4. MORPHOLOGY FILTERING
#############################
if anisotropy > 4:
    #4.a Area closing
    endo_vessel_close = np.stack([binary_closing(slice, footprint=ellipse(2,4)) for slice in endo_vessel_thresh])
    #4.b Removing small objects
    endo_vessel_no_small = remove_small_objects(endo_vessel_close, min_size=100, connectivity=2)
    #4.c Erosion
    endo_vessel_bin = np.stack([binary_erosion(slice, footprint=disk(1)) for slice in endo_vessel_no_small])

#4.d Extract metrics: area occupied

#############################
# 5. SKELETONIZATION
#############################

def skeletons_to_volume(skeletons, shape, anisotropy):
    """Transform Kkimimaro.skeleton dictionnary into numpy stack (np.ndarray(z,y,x))"""

    vol = np.zeros(shape, dtype=np.uint8)

    for skel in skeletons.values():
        verts = skel.vertices/ anisotropy
        verts = np.round(verts).astype(int)

        for v1, v2 in skel.edges:
            p1 = verts[v1]
            p2 = verts[v2]

            rr = line_nd(p1, p2)
            vol[rr] = 1

    return vol

#Skeleton slide by slide
if anisotropy > 4:
    skeleton = np.stack([skeletonize(slice) for slice in endo_vessel_bin])
    #skeleton_kimi = kimimaro.skeletonize(label(endo_vessel_bin), anisotropy=(z_pxl2um, xy_pxl2um, xy_pxl2um))
    #skeleton_stack = skeletons_to_volume(skeleton_kimi, endo_vessel_bin.shape, (z_pxl2um, xy_pxl2um, xy_pxl2um))

skel_proj = np.zeros(dims, dtype=np.uint8)
#skel_proj_kimi = np.zeros(dims, dtype=np.uint8)
for i in range(depth): 
    skel_proj += skeleton[i,:,:]
    #skel_proj_kimi += skeleton_stack[i,:,:]


#############################
# DISPLAY
#############################
fig, ax = plt.subplots(nrows=2, ncols=4)
ax[0, 0].imshow(endo_stack[0,:,:])
ax[0, 1].imshow(endo_stack_contrast[0,:,:])
ax[0, 2].imshow(endo_vessel_sato[0,:,:])
ax[0, 3].imshow(endo_vessel_thresh[0,:,:])
ax[1, 0].imshow(endo_vessel_close[0,:,:])
ax[1, 1].imshow(endo_vessel_bin[0,:,:])
ax[1, 2].imshow(skel_proj)

plt.show()