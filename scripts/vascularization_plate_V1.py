import _bootstrap
import os
import src.vessel_methods as vm
from utils.logging_setup import setup_logging
import logging
from skimage.filters import threshold_otsu, sato
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage import binary_closing
import numpy as np
import pandas as pd
import kimimaro
VERSION = "V1"
VESSEL_DIAM = 10 #in µm
CELL_RADIUS = 4 #in µm
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"{os.path.basename(__file__)} | Version={VERSION}")
logger.info("\n=================================\n      MICRO-VESSEL ANALYSIS      \n=================================")

#############################
# 0. USER INPUTS
#############################

day_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal"
out_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal-analysis"
same_result_file = True #Set True if you want to add mulitple analysis to the same file esle set False
result_file_prefix = "Exp004-E_"
filename = "Exp004-E_F5_Seqlong_x10_GFP-BF-Cy3-DAPI_16bit-1024-n4-s7_D14_2025_12_01__16_48_47.czi" #Don't put the extension '.czi' if czi_mode is 'single'
condname = "Seqshort-E5"
fluo_endo = "GFP"
fluo_chans = ["GFP","DAPI","CY3","BF"]
z_start = 1
z_end = -1 #set to -1 if you want to go until the end of the stack
czi_mode = 'stack' # 'single': if one czi file per z, 'stack': if one czi file for entire stack
logger.info("\n---------------------------------\n      User Inputs      \n---------------------------------")
logger.info(f"User Inputs |\n\tpath={day_path}\n \
    \tout_path={out_path}\n \
    \tfilename={filename}\n \
    \tfluo_chans={fluo_chans}\n \
    \tfluo_endo={fluo_endo}\n \
    \tz_start={z_start} ; z_end={z_end}\n \
    \tczi_mode={czi_mode}")

# #############################
# # 1. LOADING
# #############################
logger.info("\n---------------------------------\n      Image Loading      \n---------------------------------")

#Load stack and metadata
full_stack, scaling, metadata = vm.load_stack_metadata(day_path, filename, czi_mode)

# 1.b Keep Endo image
dims = (full_stack.shape[2], full_stack.shape[3])
endo_stack = full_stack[fluo_chans.index(fluo_endo), z_start:z_end, :, :]
depth = endo_stack.shape[0]
del full_stack
# np.save("endo_stack.npy", endo_stack)

logger.info(f"Loading Image | Stack dimension (Z,Y,X)=({depth}, {dims[0]}, {dims[1]})")
logger.info(f"Loading Image | Resolution (pxl -> um): X-Ypxl={scaling[1]}um; Zpxl={scaling[0]}um")

# ##########################################################
# # 2. PRE-PROCESSING: depends on anisotropy
# ##########################################################
logger.info("\n---------------------------------\n      Pre-Processing      \n---------------------------------")

#Determine voxel anisotropy
anisotropy = scaling[0] / scaling[1] # Z_scaling / XY_scaling
logger.info(f"Pre-Processing | Voxel anisotropy={anisotropy}")
if anisotropy > 5: 
    logger.warning("Pre-Processing | High anisotropy, all geometry data need to be interpeted carefully.\n \
                   Please condiser interpreting topological data instead.")

#Resilce to get isotropic voxels
if anisotropy != 1:
    endo_stack_iso = vm.get_isotropic_stack(endo_stack, scaling)
else:
    endo_stack_iso = endo_stack.copy()
np.save(os.path.join(out_path, (result_file_prefix + condname + "_iso.npy")), endo_stack_iso)

#Update dimension and depth
dims = endo_stack_iso.shape
depth = dims[0]

logger.info(f"Pre-Processing | Stack resliced: dims={endo_stack_iso.shape}")

#Improve contrast (usefull for uneven illumination)
endo_stack_contrast = equalize_adapthist(endo_stack_iso)
logger.info(f"Pre-Processing | Improve local contrast using CLAHE (kernel_size=1/8width, clip_lim=0.01, nbins=256)")
# np.save(os.path.join(out_path, (result_file_prefix + condname + "_contrast.npy")), endo_stack_contrast)

#Vessel Enhancement (sato seems better than frangi) --> REALLY LONG STEP !!!
vessel_diam = range(1, VESSEL_DIAM)
vessel_diam_pxl = [round(elt/(2*scaling[1]), 1) for elt in vessel_diam]
logger.info(f"Pre-Prcoessing | Enchance tubes for range (µm): {vessel_diam}\n\trange in pxl: {vessel_diam_pxl}")
endo_vessel_sato = sato(endo_stack_contrast, sigmas=vessel_diam_pxl, black_ridges=False)
np.save(os.path.join(out_path, (result_file_prefix + condname + "_sato.npy")), endo_vessel_sato)

# ##########################################################
# # 3. THRESHOLD
# ##########################################################

logger.info("\n---------------------------------\n     Threshold     \n---------------------------------")

otsu_thresh = threshold_otsu(endo_vessel_sato)
endo_vessel_thresh = endo_vessel_sato > otsu_thresh
# np.save(os.path.join(out_path, (result_file_prefix + condname + "_thresh.npy")), endo_vessel_thresh)

logger.info(f"Threshold | Otsu method: {otsu_thresh}")


# ##########################################################
# # 4. MORPHOLOGY FILTERING
# ##########################################################

logger.info("\n---------------------------------\n     Morphology Filtering     \n---------------------------------")

def generate_ellipsoid_structure(rx, ry, rz):
    """
    Create a 3D ellipsoidal structuring element.

    rx, ry, rz = radii along x, y, z
    """
    z, y, x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]

    ellipsoid = (x**2 / rx**2 +
                 y**2 / ry**2 +
                 z**2 / rz**2) <= 1

    return ellipsoid.astype(np.uint8)

#dilation, closing removing small
cell_volume = 4*np.pi*np.pow(CELL_RADIUS, 3)/3
ellipse_1_2 = generate_ellipsoid_structure(1,2,1)
ellipse_2_4 = generate_ellipsoid_structure(2,4,2)
endo_vessel_close = binary_closing(endo_vessel_thresh, ellipse_2_4)
endo_vessel_bin = remove_small_objects(endo_vessel_close, min_size=cell_volume/np.pow(scaling[1],3), connectivity=2)
del endo_vessel_close 
np.save(os.path.join(out_path, (result_file_prefix + condname + "_bin.npy")), endo_vessel_bin)

logger.info("Morphology Filtering | close (2,4,2) & remove small objects < Cell volume (~523 µm3)")

##########################################################
# 5. SKELETONIZATION
##########################################################

logger.info("\n---------------------------------\n     Skeletonization     \n---------------------------------")

teasar_param = kimimaro.intake.DEFAULT_TEASAR_PARAMS.copy()
teasar_param.update({
    "scale": 3, #defines skeleton detail (high value for less detail). Default is 1.5
    "const": 0, # control path pruning during TEASAR algo. Default is 300
    "pdrf_exponent": 8, # how much branch are penalized if close to edge of object (low values = low penalty). Default are 100000 and 4.
    })
skeleton_kimi = kimimaro.skeletonize(label(endo_vessel_bin), anisotropy=(scaling[1],scaling[1],scaling[1]), dust_threshold=pow(VESSEL_DIAM,3), fix_branching=False, teasar_params=teasar_param)
skeleton_stack = vm.skeletons_to_volume(skeleton_kimi, endo_vessel_bin.shape, (scaling[1],scaling[1],scaling[1]))
np.save(os.path.join(out_path, (result_file_prefix + condname + "_skeleton.npy")), skeleton_stack)

logger.debug(f"Skeletonization | anisotropy={scaling}, dust_threshold={pow(VESSEL_DIAM,3)},\n \
             teasar_params={teasar_param}")

##########################################################
# 6. METRICS EXTRACTION
##########################################################

logger.info("\n---------------------------------\n     Metrics Extraction     \n---------------------------------")

brenches_data = None
structures_data = None
n_skel = len(skeleton_kimi)
updated_skeleton = {}

#For each structure
for i, (skel_id, skel) in enumerate(skeleton_kimi.items()):

    print(f"Skeleton {i+1}/{n_skel}", end="\r")

    #Create networkx Graph
    G = vm.skeleton_to_graph(skel)

    #Extract brencges data
    brenches = vm.get_brenches_data(G)
    brenches["Id"] = skel_id
    brenches["Image name"] = filename
    brenches["Condition"] = condname

    #Extract strcuture data
    structure, G_nosprout = vm.get_structure_data(G, brenches, scaling[1])
    structure["Id"] = skel_id
    structure["Image name"] = filename
    structure["Condition"] = condname

    #Update skeleton
    updated_skeleton[skel_id] = vm.FakeSkeleton([ G_nosprout.nodes[elt]["coord"] for elt in G_nosprout.nodes ], #vertices
                                                G_nosprout.edges, #edges
                                                [ G_nosprout.nodes[elt]["radius"] for elt in G_nosprout.nodes ]) #radii

    #Concatenate data with previous strcutures
    if brenches_data is None:
        brenches_data = brenches
    else:
        brenches_data = pd.concat([brenches_data, brenches], ignore_index=True)
    if structures_data is None:
        structures_data = structure
    else:
        structures_data = pd.concat([structures_data, structure], ignore_index=True)

##########################################################
# 6. SAVING DATA
##########################################################

logger.info("\n---------------------------------\n     Saving Data     \n---------------------------------")
logger.info(f"Saving Data | Saving path: {out_path}")

brench_name = result_file_prefix + condname + "Brenches_Results.csv"
brenches_data.to_csv(os.path.join(out_path, result_file_prefix + condname + "Brenches_Results.csv"), index=False)
logger.info(f"Saved brenches data as {brench_name}")

structure_name = result_file_prefix + condname + "Structure_Results.csv"
structures_data.to_csv(os.path.join(out_path, structure_name), index=False)
logger.info(f"Saved strcuture data as {structure_name}'")

updated_skeleton_stack = vm.skeletons_to_volume(updated_skeleton, dims, scaling[1])
np.save(os.path.join(out_path, (result_file_prefix + condname + "_skeleton_updated.npy")), updated_skeleton_stack)

