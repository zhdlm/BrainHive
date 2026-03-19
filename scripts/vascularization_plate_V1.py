import _bootstrap
import os
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from scipy.ndimage import gaussian_filter, median_filter
from skimage.transform import rescale
from skimage.filters import threshold_otsu, threshold_yen, rank, threshold_local, threshold_multiotsu, sato
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_erosion, remove_small_objects, binary_closing, binary_opening, skeletonize, binary_dilation
from skimage.morphology import disk, ellipse
from skimage.measure import label
from skimage.draw import line_nd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from math import floor, pow
import numpy as np
import pandas as pd
from pprint import pprint
from natsort import natsorted
import SimpleITK  as sitk
import kimimaro
import czifile
import crackle
import json
import skan
import networkx
import napari
import time
import multiprocess as mp

VERSION = "V1"
VESSEL_DIAM = 10 #in µm
print(f"=================================")
print(f"      MICRO-VESSEL ANALYSIS      ")
print(f"=================================")

#############################
# 0. USER INPUTS
#############################

day_path = r"C:\Users\ChimieENS\Documents\Layla\Data\20260225-Exp006-E"
out_path = r"C:\Users\ChimieENS\Documents\Layla\Data\20260225-Exp006-E"
filename = "Exp006-E_Seq-1_x20_GFP_DELTAZ2UM_16bit-1024_2026_02_25__16_42_22" #Don't put the extension '.czi' if czi_mode is 'single'
fluo_endo = "GFP"
fluo_chans = ["GFP","CY3","BF","DAPI"]
z_start = 0
z_end = -1 #set to -1 if you want to go until the end of the stack
czi_mode = 'single' # 'single': if one czi file per z, 'stack': if one czi file for entire stack
print("\n--------------------------------")
print("\tUSER INPUTS")
print(f"\tpath={day_path}\n \
      \tout_path={out_path}\n \
      \tfilename={filename}\n \
      \tfluo_chans={fluo_chans}\n \
      \tfluo_endo={fluo_endo}\n \
      \tz_start={z_start} ; z_end={z_end}\n \
      \tczi_mode={czi_mode}")
print("--------------------------------")

#############################
# 1. LOADING
#############################

#1.a Load stack and metadata
#---------------------------

if czi_mode == 'single':
    #Store filenames
    files = os.listdir(day_path)
    filenames = [f for f in files if filename in f]
    filenames = natsorted(filenames)
    if "(" not in filenames[-1]:
        filenames.insert(0, filenames[-1])
        del filenames[-1]
    
    #Load metadata and save dimension
    czi = czifile.CziFile(os.path.join(day_path, filenames[0]))
    metadata = czi.metadata(asdict=True)
    dim_loc = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
    if "SizeC" in dim_loc.keys():
        dim_C = dim_loc["SizeC"]
    else:
        dim_C = 1
    dims = (dim_C, dim_loc["SizeZ"], dim_loc["SizeY"], dim_loc["SizeX"])
    full_stack = np.ndarray(dims, dtype='uint16')
    del dims
    czi.close()

    #Open files and store them in stack
    c=0
    for i,f in enumerate(filenames):
        if (i % dim_C == 0):
            c=0
        else:
            c+=1
        if i != 0 and ((i+1) % dim_C == 0):
            z_i = floor((i)/dim_C)
        else:
            z_i = floor((i+1)/dim_C)
        czi = czifile.CziFile(os.path.join(day_path, f))
        full_stack[c,z_i,:,:] = czi.asarray()
        czi.close()


elif czi_mode == 'stack':
    czi = czifile.CziFile(os.path.join(day_path, filename)) # CZI file + metadata
    full_stack = czi.asarray() #Full stack of shape (Channel, Z, Y, X)
    metadata = czi.metadata(asdict=True)

# Store voxel value
pxls2um = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
xy_pxl2um = pxls2um[0]["Value"]*1000000
z_pxl2um = pxls2um[2]["Value"]*1000000
z_step_um = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Z"]["Positions"]["Interval"]["Increment"]


#1.b Keep Endo image
#-------------------

dims = (full_stack.shape[2], full_stack.shape[3])
endo_stack = full_stack[fluo_chans.index(fluo_endo), z_start:z_end, :, :]
depth = endo_stack.shape[0]
del full_stack
np.save("endo_stack.npy", endo_stack)

print("\n--------------------------------")
print("\tLOADING IMAGE")
print(f"Stack dimension (Z,Y,X)=({depth}, {dims[0]}, {dims[1]})")
print(f"Resolution (pxl -> um): X-Ypxl={round(xy_pxl2um,1)}um; Zpxl={z_pxl2um}um")
print("--------------------------------")

#1.c Metrics extraction: depth, I_mean_raw(z), I_std_raw(z), I_med_raw(z), I_mad_raw(z)
#--------------------------------------------------------------------------------------

##########################################################
# 2. PRE-PROCESSING: depends on anisotropy
##########################################################

#2.a Reslice stack to compensate anisotropy
#----------------
anisotropy = z_pxl2um / xy_pxl2um
print("\n--------------------------------")
print("\tPRE-PROCESSING\n")
print(f"Voxel anisotropy={anisotropy}")
if anisotropy > 5: print("WARNING: High anisotropy, all geometry data need to be interpeted carefully. Please condiser interpreting topological data instead.")

if anisotropy != 1:
    img = sitk.GetImageFromArray(endo_stack)
    img.SetSpacing((xy_pxl2um, xy_pxl2um, z_pxl2um))

    original_size = np.array(img.GetSize())
    original_spacing = np.array(img.GetSpacing())

    new_spacing = np.array([xy_pxl2um, xy_pxl2um, xy_pxl2um])
    new_size = (original_size * (original_spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(img)

    endo_stack_iso = sitk.GetArrayFromImage(resampler.Execute(img))
    dims = endo_stack_iso.shape
    depth = dims[0]
    print(f"Stack resliced: dims={dims}")
    del img, resampler, new_spacing
else:
    endo_stack_iso = endo_stack.copy()
np.save("endo_stack_iso.npy", endo_stack_iso)



#2.b Improve contrast (usefull for uneven illumination)
#------------------------------------------------------

endo_stack_contrast = equalize_adapthist(endo_stack_iso)
np.save("endo_stack_contrast.npy", endo_stack_contrast)
# np.save("endo_stack_contrast.npy", endo_stack_contrast)
print(f"Improve local contrast using CLAHE (kernel_size=1/8width, clip_lim=0.01, nbins=256)")

#2.c Vessel Enhancement (sato seems better than frangi) --> REALLY LONG STEP !!!
#------------------------------------------------------

vessel_diam = range(2, 15)
vessel_diam_pxl = [round(xy_pxl2um*elt/2, 1) for elt in vessel_diam]
print(f"Vessel Diameter searched (µm): {vessel_diam}\n\tsearched in pxl: {vessel_diam_pxl}")
print("Vessel enhancement. WARNING: This step is quite long to execute.")
start = time.time()
endo_vessel_sato_optisigma2_15 = sato(endo_stack_contrast, sigmas=vessel_diam_pxl, black_ridges=False)
np.save("endo_vessel_sato_1-15um.npy", endo_vessel_sato_optisigma2_15)
del endo_vessel_sato_optisigma2_15

quit()

runtime = time.time() - start
print(f"Sato runtime={runtime}")
print("--------------------------------")

##########################################################
# 3. THRESHOLD
##########################################################

otsu_thresh = threshold_otsu(endo_vessel_sato)
endo_vessel_thresh = endo_vessel_sato > otsu_thresh
np.save("endo_vessel_thresh.npy", endo_vessel_thresh)
print("\n--------------------------------")
print("\tTHRESHOLD")
print(f"Otsu Thesrhold: otsu={otsu_thresh}")
print("--------------------------------")


##########################################################
# 4. MORPHOLOGY FILTERING
##########################################################

#4.a Filter mask based on morphology
#-----------------------------------

#Area closing
endo_vessel_close = np.stack([binary_closing(slice, footprint=ellipse(2,4)) for slice in endo_vessel_thresh])
#Removing small objects
endo_vessel_no_small = remove_small_objects(endo_vessel_close, min_size=100, connectivity=2)
#Erosion
endo_vessel_erosion = np.stack([binary_erosion(slice, footprint=disk(1)) for slice in endo_vessel_no_small])
np.save("endo_vessel_cl_no_ero.npy", endo_vessel_erosion)
np.save("endo_vessel_cl_no.npy", endo_vessel_no_small)

#dilation, closing removing small
endo_vessel_dil = np.stack([binary_dilation(slice, footprint=ellipse(1,2)) for slice in endo_vessel_thresh])
endo_vessel_close = np.stack([binary_closing(slice, footprint=ellipse(2,4)) for slice in endo_vessel_dil])
endo_vessel_no_small = remove_small_objects(endo_vessel_close, min_size=100, connectivity=2)
endo_vessel_erosion = np.stack([binary_erosion(slice, footprint=disk(2)) for slice in endo_vessel_no_small])
np.save("endo_vessel_dil_cl_no_ero.npy", endo_vessel_erosion)
np.save("endo_vessel_dil_cl_no.npy", endo_vessel_no_small)

quit()

print("\n--------------------------------")
print("\tMORPHOLOGY FILTERING\nclosing (2,4) > remove small objects > erosion (1)")
print("--------------------------------")

#4.b Extract metrics: area occupied (total and per slice)
#--------------------------------------------------------

##########################################################
# 5. SKELETONIZATION
##########################################################

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

    
skeleton_kimi = kimimaro.skeletonize(label(endo_vessel_bin), anisotropy=(z_pxl2um, xy_pxl2um, xy_pxl2um))
skeleton_stack = skeletons_to_volume(skeleton_kimi, endo_vessel_bin.shape, (z_pxl2um, xy_pxl2um, xy_pxl2um))
np.save("skeleton_stack.npy", skeleton_stack)

skel_proj_kimi = np.zeros(dims, dtype=np.uint8)
for i in range(depth): 
    skel_proj_kimi += skeleton_stack[i,:,:]

print("\n--------------------------------")
print("\tSKELETON")
print("give some info, which ones ?")
print("--------------------------------")

##########################################################
# 6. METRICS EXTRACTION
##########################################################

def c_network_coverage(mask, scaling):
    
    dims = mask.shape
    volume_tot_pxl = np.count_nonzero(mask)
    volume_tot_um = volume_tot_pxl*pow(scaling)
    volume_coverage = volume_tot_pxl/(dims[0]*dims[1]*dims[2])

    return volume_tot_um, volume_coverage


headers_global = [
    #Low sensitivity to anisotropy (safe for anisotropy <= 6)
    "Network area coverage",
    "Branch density",
    "Junction density",
    "Node density",
    "Diameter (XY-oriented)",
    "Vessel tortuosity",
    "Segment orientation",
    "Structural anisotropy",
    "Vessel volume",
    "3D branch density",
    "Pore size",
    "Pore size distribution",
    "Number of sprouts",
    "Radial invasion distance",
    #Med sensitivity to anisotropy (safe for anisotropy <= 4-5)
    "Mean branch length",
    "Loop density",
    "Mean segment length (3D)",
    "Total tube length",
    "Number of branches",
    "Number of junctions",
    "Number of nodes",
    "Number of endpoints",
    "Mesh / loop count",
    "Mesh area",
    "Diameter variability",
    "Vessel surface area",
    "Surface-to-volume ratio",
    "Lumen volume",
    "Sprout length",
    #High sensitivity to anisotropy (safe for anisotropy <=2-3)
    "Vessel diameter",
    "Vessel thickness",
    "Lumen diameter"

]


data_global = { key: None for key in headers_global}
print(data_global)

print("\n--------------------------------")
print("\tMETRICS EXTRACTION")

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
ax[1, 2].imshow(skel_proj_kimi)
#ax[1, 3].imshow(skel_proj_kimi)
plt.show()

