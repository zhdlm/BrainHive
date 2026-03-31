import os
from pprint import pprint
import czifile
from natsort import natsorted
import numpy as np
from math import floor
import SimpleITK  as sitk
import networkx as nx
import pandas as pd
from osteoid import Skeleton
from utils.usefull_functions import flatten_list_recursive
from skimage.filters import threshold_otsu, sato
from skimage.morphology import label, remove_small_objects
from skimage.exposure import equalize_adapthist
from skimage.draw import line_nd, draw3d
from scipy.ndimage import gaussian_filter
from skimage import img_as_ubyte
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation
import napari
import kimimaro
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import csv
import logging
logger = logging.getLogger(__name__)

#For tests
class FakeSkeleton:
    def __init__(self, vertices, edges, radii, segid=None):
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        self.radii = np.array(radii)
        self.id = segid

def create_microvessel_stack():
    """
    ```
    top view                (diago)
                          8 ●
                           /     
                        7 ●
                       4 /   
        ● — ● — ● — ● — ● — ● — ● (main)
        0   1   2   3   |   5   6
                  \     ● 9
                    ●   |
                  12  \ ● 10
                        |
                        ● 11
                        (bottom)
    ```
    """

    vertices = [
        (30,50, 5), (30,50, 30), (40,50,40), (60,50,50), (60,50,60), (40,50,80), (40,50,90), #main
        (60,60,60), (60,70,70), # diago
        (60,40,50), (50,30,50), (40,20,50), #bottom
        (40,40,20) #loop
    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), #main
        (4,7), (7,8),
        (4,9), (9,10), (10,11),
        (2,12), (11,12)
    ]
    
    radii = [2.9, 3, 3.1, 4, 5, 4, 3.5, 
             3, 2.5,
             3.5, 3.1, 3, 
             2
    ]
    
    vertices = np.array(vertices, dtype='uint8')
    edges = np.array(edges, dtype='uint8')
    radii = np.array(radii, dtype='uint8')

    # --- Create 3D grid ---
    margin = int(np.max(radii)) + 2

    grid = np.zeros((100,100,100), dtype=np.uint8)

    # --- Helper: draw sphere ---
    def draw_circle(grid, center, radius):
        x0, y0, z0 = center.astype(int)
        r = int(np.ceil(radius))
        z = set()
        for x in range(x0 - r, x0 + r + 1):
            for y in range(y0 - r, y0 + r + 1):
                for z in range(z0 - r, z0 + r + 1):
                    if (
                        0 <= x < grid.shape[0] and
                        0 <= y < grid.shape[1] and
                        0 <= z < grid.shape[2]
                    ):
                        if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                            val = 100 - z/100
                            # print(val)
                            grid[x, y, z] = 100 - z/10

    # --- Draw tubes ---
    for i, j in edges:
        p0 = vertices[i]
        p1 = vertices[j]
        r0 = radii[i]
        r1 = radii[j]

        length = np.linalg.norm(p1 - p0)
        steps = int(length * 2) + 1  # sampling density

        for t in np.linspace(0, 1, steps):
            p = (1 - t) * p0 + t * p1
            r = (1 - t) * r0 + t * r1
            draw_circle(grid, p, r)

        #Apply blur
        grid = gaussian_filter(grid, 0.5)
    
    print(type(grid), grid.shape, grid.min(), grid.max())

    return grid

    def world_to_voxel(p):
        return np.array([
            p[0] / vz,
            p[1] / vy,
            p[2] / vx
        ])

    def draw_sphere(center_vox, radius_phys, intensity):

        rz = radius_phys / vz
        ry = radius_phys / vy
        rx = radius_phys / vx

        cz, cy, cx = center_vox

        zmin = max(0, int(cz - rz - 1))
        zmax = min(Z, int(cz + rz + 2))
        ymin = max(0, int(cy - ry - 1))
        ymax = min(Y, int(cy + ry + 2))
        xmin = max(0, int(cx - rx - 1))
        xmax = min(X, int(cx + rx + 2))

        for z in range(zmin, zmax):
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    dz = (z - cz) * vz
                    dy = (y - cy) * vy
                    dx = (x - cx) * vx

                    if dz*dz + dy*dy + dx*dx <= radius_phys**2:
                        vol[z, y, x] = max(vol[z, y, x], intensity)

    for e_idx, (i, j) in enumerate(edges):
        p0 = np.array(vertices[i])
        p1 = np.array(vertices[j])

        r = radii[e_idx]

        # intensity linked to radius
        if radius_intensity:
            intensity = int(60 + (r / max(radii)) * 180)
        else:
            intensity = base_intensity

        # convert to voxel space
        v0 = world_to_voxel(p0)
        v1 = world_to_voxel(p1)

        length = np.linalg.norm((v1 - v0) * np.array([vz, vy, vx]))

        # sampling density (important!)
        step = max(r / 2, 0.5)
        n_samples = max(2, int(length / step))

        for t in np.linspace(0, 1, n_samples):
            p = v0 * (1 - t) + v1 * t
            draw_sphere(p, r, intensity)

    return vol

def create_test_skeleton_no_loop():
    """
    ```
            (branch) 8
            |
            ● 7
            |
    ● — ● — ● — ● — ● — ● — ●
    0   1   |2  3   4   5   6
          9 ●
            |
         10 ●
            |
         11 (branch)

    ```
    """
    vertices = [
        (-1, 0, 0),   # 0
        (0, 0, 1),   # 1
        (0, 0, 2),   # 2 ← branch point
        (0, 0, 3),   # 3
        (0, 0, 4),   # 4
        (0, 0, 5),   # 5
        (0, 0, 6),   # 6

        (0, 1, 2),   # 7 (branch up)
        (0, 2, 2),   # 8

        (0, -1, 2),  # 9 (branch down)
        (0, -2, 2),  # 10
        (0, -3, 2)   # 11
    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
        (2,7), (7,8),
        (2,9), (9,10), (10,11)
    ]

    # Radii per node
    radii = [
        1.0,  # 0
        1.1,  # 1
        1.2,  # 2 (slightly larger at junction)
        1.1,  # 3
        1.0,  # 4
        1.1,  # 5
        1.0,  # 6

        0.8,  # 7
        0.7,  # 8

        0.8,  # 9
        0.7,  # 10
        0.7  # 11
    ]

    return FakeSkeleton(vertices, edges, radii)

def create_test_skeleton_loop():
    """
    ```
         8  x
            |   13
         7  x — ● — ● 12 loop
            |       |
    x — ● — x — ● — x — ● — x
    0   1   |2  3   4   5   6
          9 ●
            |
         10 ●
            |
         11 x

    ```
    """
    vertices = [
        (-1, 0, 0),   # 0
        (0, 0, 1),   # 1
        (0, 0, 2),   # 2 ← branch point
        (0, 0, 3),   # 3
        (0, 0, 4),   # 4
        (0, 0, 5),   # 5
        (0, 0, 6),   # 6

        (0, 1, 2),   # 7 (branch up)
        (0, 2, 2),   # 8

        (0, -1, 2),  # 9 (branch down)
        (0, -2, 2),  # 10
        (0, -3, 2),  # 11

        (0, 1, 4),   # 12 (loop)
        (0, 1, 3)    # 13   



    ]

    edges = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
        (2,7), (7,8),
        (2,9), (9,10), (10,11),
        (4,12), (12,13), (7,13)
    ]

    # Radii per node
    radii = [
        1.0,  # 0
        1.1,  # 1
        1.2,  # 2 (slightly larger at junction)
        1.1,  # 3
        1.0,  # 4
        1.1,  # 5
        1.0,  # 6

        0.8,  # 7
        0.7,  # 8

        0.8,  # 9
        0.7,  # 10
        0.7,  # 11

        0.7,  # 12
        0.7,  # 13
    ]

    return FakeSkeleton(vertices, edges, radii)

#For real data

def read_usr_inputs(csv_filename):
    
    if os.path.isfile(csv_filename) == False:
        # logger.error(f"Input csv file not found : {csv_filename}")
        raise ValueError(f"{csv_filename} not found.")

    user_inputs = {"Params": []}
    headers = []
    f = open(csv_filename, 'r', newline='', encoding='utf-8-sig')
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for r, row in enumerate(reader):
        if "".join(row) == "":
            idx_empty_row = r
            break
    f.seek(0) #go back at begining of document
    for r, row in enumerate(reader):
        if r < idx_empty_row:
            user_inputs.update({row[0]: row[1]})
        elif r == idx_empty_row + 1:
            for col in row:
                headers.append(col)
                user_inputs.update({col: []})
        elif r > idx_empty_row + 1:
            user_inputs["Params"].append({})
            for c, col in enumerate(row):
                if ";" in col:
                    col = col.split(";")
                user_inputs["Params"][-1].update({headers[c]: col})
                user_inputs[headers[c]].append(col)

    logger.info("\n---------------------------------\n      User Inputs      \n---------------------------------")
    logger.info(f"User Inputs |\n\tpath={user_inputs["day_path"]}\n \
    \tout_path={user_inputs}\n \
    \tfilename={user_inputs["filename"]}\n \
    \tcondname={user_inputs["condname"]}\n \
    \tfluo_chans={user_inputs["fluo_chans"]}\n \
    \tfluo_endo={user_inputs["fluo_endo"]}\n \
    \tz_start={user_inputs["z_start"]} ; z_end={user_inputs["z_end"]}\n \
    \tczi_mode={user_inputs["czi_mode"]}\n \
    ")

    return user_inputs

def load_fluo_stack(usr, index, save=False):
    """Return the scaling of the stack and the stack of endothelial cells as ndarray of dimensions 
    (Z,Y,X) for the slices requested by user and the fluorescent channel labelling endothelial cells.
    
    Parameters
    ----------
    usr: dict
        dictionnary, returned by read_usr_inputs method, summarizing all parameters enetered by user and
        requested to perform microvessel analysis.
    index: int
        index of the elements to use in usr variable.

    Returns
    -------
    endo_stack: numpy.ndarray
        Stack of the microvessels (on channel `usr["fluo_endo"]`) of dimension (Z,Y,X) for the slices from
        `z_start` to `z_end`.
    scaling: tuple
        Scaling of the stack, ie micrometer length of one pixel, for each dimension (z_scaling,
        y_scaling, x_scaling)
    """

    #Logs on current image info
    logger.info("\n---------------------------------\n      Image Loading      \n---------------------------------")
    logger.info(f"Loading Image | User inputs:\n \
                \tpath={usr["day_path"]}\n \
                \tout_path={usr["out_path"]}\n \
                \tfilename={usr["filename"][index]}\n \
                \tcondname={usr["condname"][index]}\n \
                \tfluo_chans={usr["fluo_chans"][index]}\n \
                \tfluo_endo={usr["fluo_endo"][index]}\n \
                \tz_start={usr["z_start"][index]} ; z_end={usr["z_end"][index]}\n \
                \tczi_mode={usr["czi_mode"][index]}\n \
                ")
    
    #Load entire stack & metadata
    full_stack, scaling, metadata = load_stack_metadata(usr["day_path"], usr["filename"][index], usr["czi_mode"][index])

    #Keep only fluorescent channel labelling endothelial cells for slices requested by user
    endo_stack = full_stack[usr["fluo_chans"][index].index(usr["fluo_endo"][index]), int(usr["z_start"][index]):int(usr["z_end"][index]), :, :]

    #remove unsued variables
    del full_stack, metadata

    #Logs on output
    logger.info(f"Loading Image | Stack dimension (Z,Y,X)={endo_stack.shape}")
    logger.info(f"Loading Image | Resolution (pxl -> um): X-Ypxl={scaling[1]}um; Zpxl={scaling[0]}um")

    return endo_stack, scaling

def load_stack_metadata(day_path: str, filename: str, czi_mode: str):
    """Returns the full stack as ndarray (C,Z,Y,X), the associated metadata adn the scaling (Z,Y,X)"""

    full_stack = None
    scaling = None
    metadata = None

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
        metadata = czi.metadata(asdict=True)
        dim_loc = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
        stack = czi.asarray() #Full stack of shape (Channel, Z, Y, X)
        if "SizeC" in dim_loc.keys():
            dim_C = dim_loc["SizeC"]
            full_stack = stack
        else:
            dim_C = 1
            dims = (dim_C, dim_loc["SizeZ"], dim_loc["SizeY"], dim_loc["SizeX"])
            full_stack = np.ndarray(dims, dtype='uint16')
            full_stack[0,:,:,:] = stack


    # Store voxel value
    pxls2um = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    scaling = (pxls2um[2]["Value"]*1000000, pxls2um[0]["Value"]*1000000, pxls2um[0]["Value"]*1000000)
    z_step_um = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Z"]["Positions"]["Interval"]["Increment"]

    return full_stack, scaling, metadata

def pre_processing(usr: dict, f: int, endo_stack: np.ndarray, scaling: tuple, VESSEL_DIAM: float):

    logger.info("\n---------------------------------\n      Pre-Processing      \n---------------------------------")

    #Determine voxel anisotropy
    anisotropy = scaling[0] / scaling[1] # Z_scaling / XY_scaling
    logger.info(f"Pre-Processing | Voxel anisotropy={anisotropy}")
    if anisotropy > 5: 
        logger.warning("Pre-Processing | High anisotropy, all geometry data need to be interpeted carefully.\n \
                       Please condiser interpreting topological data instead.")

    #Resilce to get isotropic voxels
    if anisotropy != 1:
        endo_stack_iso = get_isotropic_stack(endo_stack, scaling)
    else:
        endo_stack_iso = endo_stack.copy()
    np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][f] + "_iso.npy")), endo_stack_iso)

    #Update dimension and depth
    dims = endo_stack_iso.shape

    logger.info(f"Pre-Processing | Stack resliced: dims={endo_stack_iso.shape}")

    #Improve contrast (usefull for uneven illumination)
    endo_stack_contrast = equalize_adapthist(endo_stack_iso)
    logger.info(f"Pre-Processing | Improve local contrast using CLAHE (kernel_size=1/8width, clip_lim=0.01, nbins=256)")
    del endo_stack_iso
    # np.save(os.path.join(out_path, (exp_name + condname + "_contrast.npy")), endo_stack_contrast)

    #Vessel Enhancement (sato seems better than frangi) --> REALLY LONG STEP !!!
    vessel_diam = range(1, VESSEL_DIAM)
    vessel_diam_pxl = [round(elt/(2*scaling[1]), 1) for elt in vessel_diam]
    logger.info(f"Pre-Prcoessing | Enchance tubes for range (µm): {vessel_diam}\n\trange in pxl: {vessel_diam_pxl}")
    endo_vessel_sato = sato(endo_stack_contrast, sigmas=vessel_diam_pxl, black_ridges=False)
    np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][f] + "_sato.npy")), endo_vessel_sato)
    del endo_stack_contrast

    return endo_vessel_sato

def get_isotropic_stack(stack, scaling: tuple, save=False):
    """Return resliced stack as np.ndarray of dimension (Z,Y,X) with isotropic voxel."""
    
    img = sitk.GetImageFromArray(stack)
    img.SetSpacing((scaling[1], scaling[1], scaling[0]))

    original_size = np.array(img.GetSize())
    original_spacing = np.array(img.GetSpacing())

    new_spacing = np.array([scaling[1], scaling[1], scaling[1]])
    new_size = (original_size * (original_spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(img)

    stack_iso = sitk.GetArrayFromImage(resampler.Execute(img))
    del img, resampler, new_spacing

    return stack_iso

def threshold(stack):

    logger.info("\n---------------------------------\n     Threshold     \n---------------------------------")

    otsu_thresh = threshold_otsu(stack)
    thresh = stack > otsu_thresh
    del stack
    # np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][f] + "_thresh.npy")), endo_vessel_thresh)

    logger.info(f"Threshold | Otsu method: {otsu_thresh}")

    return thresh

def morpho_filter(usr, index, stack_thresholded: np.ndarray, scaling, CELL_RADIUS: float):

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
    endo_vessel_dil = binary_dilation(stack_thresholded, ellipse_1_2)
    endo_vessel_close = binary_closing(endo_vessel_dil, ellipse_2_4)
    endo_vessel_rm = remove_small_objects(endo_vessel_close, min_size=cell_volume/np.pow(scaling[1],3), connectivity=2)
    endo_vessel_erode = binary_erosion(endo_vessel_rm, ellipse_2_4)
    endo_vessel_bin = binary_dilation(endo_vessel_erode, ellipse_1_2)
    del endo_vessel_close, stack_thresholded
    np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][index] + "_bin.npy")), endo_vessel_bin)

    logger.info("Morphology Filtering | close (2,4,2) & remove small objects < Cell volume (~523 µm3)")

    return endo_vessel_bin

def skeletonization(usr, index, mask, scaling, save=True):
    """Return skeleton of microvsessel mask based on kimimaro.skeletonize method.

    The methods runs kimimaro.skeletonize to get the skeletons of all strcutres
    present in the stack. Then, a post-process is ran to get rid off noisy brenches.
    The post-process first re-construct the brenches then remove the ones that are
    smaller than 2 voxels or that are perfectly straight. The mask is reconstructed
    from the cleaned skeleton to repare broken brenches and the kimimaro.skeletonize
    method is run again to get final skeleton.
    
    Parameters
    ----------
    dictionnary, returned by read_usr_inputs method, summarizing all parameters enetered by user and
        requested to perform microvessel analysis.
    index: int
        index of the elements to use in usr variable.
    mask: np.ndarray
        Mask of microvessel stack of dimension (Z,Y,X).  Microvessels have value 255
        and the rest is set at 0.
    scaling: tuple
        Scaling of the stack, ie micrometer length of one pixel, for each dimension (z_scaling,
        y_scaling, x_scaling)

    Returns
    -------
    skeleton_kimi: dict[osteoid.Skeleton]
        Dictionnary where each value is a skeleton of one of the strcuture composing the mask.

    """


    logger.info("\n---------------------------------\n     Skeletonization     \n---------------------------------")

    #Determine skeletons
    logger.info("Skeletonization | Compute Skeletons")
    teasar_param = kimimaro.intake.DEFAULT_TEASAR_PARAMS.copy()
    teasar_param.update({
        "scale": 3, #defines skeleton detail (high value for less detail). Default is 1.5
        "const": 0, # control path pruning during TEASAR algo. Default is 300
        "pdrf_exponent": 8, # how much branch are penalized if close to edge of object (low values = low penalty). Default is 4.
        })
    # skeleton = kimimaro.skeletonize(label(mask), anisotropy=(scaling[1],scaling[1],scaling[1]), dust_threshold=100, fix_branching=False, teasar_params=teasar_param)
    skeleton = kimimaro.skeletonize(label(mask), anisotropy=(scaling[1],scaling[1],scaling[1]), dust_threshold=1000, fix_branching=False, teasar_params=teasar_param)

    #Clean skeletons:
    skeleton_kimi = {}
    logger.info("Skeletonization | Clean Skeletons")
    for i, skel in enumerate(skeleton.values()):

        print(f"Structure {i}/{len(skeleton)}", end="\r")

        #Create networkx graph
        G = skeleton_to_graph(skel)
        #Extract brenches info
        brenches = get_brenches_data(G, (scaling[1],scaling[1],scaling[1]))
        #Remove noisy brenches from brenches and graph
        G_updated, _, _, _, _ = brenches_post_process(G, brenches, (scaling[1], scaling[1], scaling[1]))
        #Create the skeleton without the noisy brenches
        skeleton_updated = FakeSkeleton([ G_updated.nodes[elt]["coord"] for elt in G_updated.nodes ], #vertices
                                                        G_updated.edges, #edges
                                                        [ G_updated.nodes[elt]["radius"] for elt in G_updated.nodes ],
                                                        segid=skel.id) #radii
        #Dilate the skeleton to get the clean mask of microvessels
        mask_clean = skel2mask(skeleton_updated, mask.shape, scaling)
        #Re-run skeletonization on cleaned mask
        #skel_opti = kimimaro.skeletonize(label(mask_clean), anisotropy=(scaling[1],scaling[1],scaling[1]), dust_threshold=100, fix_branching=False, teasar_params=teasar_param, progress=False)
        skel_opti = kimimaro.skeletonize(label(mask_clean), anisotropy=(scaling[1],scaling[1],scaling[1]), dust_threshold=1000, fix_branching=False, teasar_params=teasar_param, progress=False)
        #Assign the skeleton to the output dictionnary
        if len(skel_opti) == 0:
            pass
        else:
            skeleton_kimi[i] = skel_opti[1]
            skeleton_kimi[i].id = i

    print()

    #Save the clean skeleton
    if save == True:
        skeleton_stack = skeletons_to_volume(skeleton, mask.shape, (scaling[1],scaling[1],scaling[1]))
        np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][index] + f"_skeleton.npy")), skeleton_stack)
        skeleton_stack_opti = skeletons_to_volume(skeleton_kimi, mask.shape, (scaling[1],scaling[1],scaling[1]))    
        np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][index] + f"_skeleton_opti.npy")), skeleton_stack_opti)
        np.save(os.path.join(usr["out_path"], (usr["condname"][index] + "_bin2.npy")), mask_clean)
        del skeleton_stack, skeleton_stack_opti

    logger.debug(f"Skeletonization | anisotropy={scaling}, dust_threshold=100,\n \
                teasar_params={teasar_param}")
    
    del skeleton, G, brenches, mask_clean, skel_opti, skeleton_updated, G_updated
    
    return skeleton_kimi

def extract_save_metrics(usr: dict, f: int, mask: np.ndarray, skeletons: dict, scaling: tuple):
    logger.info("\n---------------------------------\n     Metrics Extraction     \n---------------------------------")

    #Mask metrics
    bin = round(20 / scaling[1])
    vessel_vol = np.count_nonzero(mask)
    stack_vol = mask.shape[0]*mask.shape[1]*mask.shape[2]
    density = 100 * vessel_vol / stack_vol
    density_bin = [100*np.count_nonzero(mask[b:b+bin, :, :])/stack_vol for b in range(0, round(400/scaling[1]), bin)]
    mask_data = pd.DataFrame({"Volume Density": density_bin})
    mask_data["Image name"] = usr["filename"][f]
    mask_data["Condition"] = usr["condname"][f]
    mask_data["Stack used"] = [(usr["z_start"][f], usr["z_end"][f])]*len(density_bin)

    #Skeleton metrics
    brenches_data = None
    structures_data = None
    updated_skeleton = {}

    #For each structure
    for i, skel in enumerate(skeletons.values()):

        # print(f"Skeleton {i+1}/{n_skel}", end="\r")
        print("id", skel.id)

        #Create networkx Graph
        G = skeleton_to_graph(skel)

        #Extract brenches data
        brenches = get_brenches_data(G, scaling)
        brenches["Id"] = skel.id
        brenches["Image name"] = usr["filename"][f]
        brenches["Condition"] = usr["condname"][f]
        brenches["Stack used"] = [(usr["z_start"][f], usr["z_end"][f])]*len(brenches)

        #Remove noise brenches from G and brenches
        G_updated, brenches_updated, _, _, _ = brenches_post_process(G, brenches, scaling)
        print(brenches_updated)

        #Extract strcuture data
        structure = get_structure_data(G, brenches, scaling[1])
        structure["Id"] = skel.id
        structure["Image name"] = usr["filename"][f]
        structure["Condition"] = usr["condname"][f]
        structure["Stack used"] = [(usr["z_start"][f], usr["z_end"][f])]*len(structure)

        #Update skeleton
        updated_skeleton[skel.id] = FakeSkeleton([ G_updated.nodes[elt]["coord"] for elt in G_updated.nodes ], #vertices
                                                    G_updated.edges, #edges
                                                    [ G_updated.nodes[elt]["radius"] for elt in G_updated.nodes ],
                                                    segid=skel.id) #radii

        #Concatenate data with previous strcutures
        if brenches_data is None:
            brenches_data = brenches
        else:
            brenches_data = pd.concat([brenches_data, brenches_updated], ignore_index=True)
        if structures_data is None:
            structures_data = structure
        else:
            structures_data = pd.concat([structures_data, structure], ignore_index=True)

    #Save data
    mask_name = usr["exp_name"] + usr["condname"][f] + "_Mask_Results.csv"
    mask_data.to_csv(os.path.join(usr["out_path"], mask_name), index=False)
    logger.info(f"Metrics Extraction | Saved mask data as {mask_name}")
    brench_name = usr["exp_name"] + usr["condname"][f] + "_Brenches_Results.csv"
    brenches_data.to_csv(os.path.join(usr["out_path"], brench_name), index=False)
    logger.info(f"Metrics Extraction | Saved brenches data as {brench_name}")
    structure_name = usr["exp_name"] + usr["condname"][f] + "_Structure_Results.csv"
    structures_data.to_csv(os.path.join(usr["out_path"], structure_name), index=False)
    logger.info(f"Metrics Extraction | Saved strcuture data as {structure_name}'")
    updated_skeleton_stack = skeletons_to_volume(updated_skeleton, mask.shape, scaling[1])
    np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][f] + "_skeleton_final.npy")), updated_skeleton_stack)

    return brenches_data, structures_data

def skeletons_to_volume(skeletons: dict[Skeleton], shape: tuple, anisotropy: tuple):
    """Transform Kimimaro.skeleton dictionnary into numpy stack (np.ndarray(z,y,x))"""

    vol = np.zeros(shape, dtype=np.uint8)

    for skel in skeletons.values():
        verts = skel.vertices / anisotropy
        verts = np.round(verts).astype(int)
        skel_id = skel.id

        for v1, v2 in skel.edges:
            p1 = verts[v1]
            p2 = verts[v2]

            rr = line_nd(p1, p2)
            vol[rr] = 1

    return vol

def skel2mask(skel, shape, anisotropy):
    
    mask = np.zeros(shape, dtype=np.uint8)

    # --- Helper: draw sphere ---
    def draw_circle(grid, center, radius):
        x0, y0, z0 = center.astype(int)
        r = int(np.ceil(radius))
        z = set()
        for x in range(x0 - r, x0 + r + 1):
            for y in range(y0 - r, y0 + r + 1):
                for z in range(z0 - r, z0 + r + 1):
                    if (
                        0 <= x < grid.shape[0] and
                        0 <= y < grid.shape[1] and
                        0 <= z < grid.shape[2]
                    ):
                        if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                            grid[x, y, z] = 1

    # --- Draw tubes ---
    for i, j in skel.edges:
        p0 = skel.vertices[i] / anisotropy
        p1 = skel.vertices[j] / anisotropy
        r0 = skel.radii[i] / anisotropy[1]
        r1 = skel.radii[j] / anisotropy[1]

        length = np.linalg.norm(p1 - p0)
        steps = int(length * 2) + 1  # sampling density

        for t in np.linspace(0, 1, steps):
            p = (1 - t) * p0 + t * p1
            r = (1 - t) * r0 + t * r1
            draw_circle(mask, p, r)

    return mask

def norm_3d(v1, v2):
    """Return length between two nodes"""
    s=0
    for i in range(len(v1)):
        s += np.pow((v2[i] - v1[i]), 2)
    return np.sqrt(s)

def skeleton_to_graph(skel: Skeleton):
    """Creates Networkx Graph object from skeleton object (osteid library) and compute metrics"""
    
    G = nx.Graph()

    # Add nodes with 3D coordinates
    for i, v in enumerate(skel.vertices):
        G.add_node(i, coord=tuple(v), radius=skel.radii[i])

    # Add edges
    for e in skel.edges:
        n1, n2 = int(e[0]), int(e[1])

        #Compute branch metrics
        p1 = skel.vertices[n1]
        p2 = skel.vertices[n2]
        length = norm_3d(p1, p2)
        direction = (p2 - p1) / norm_3d(p1, p2)
        # radius = skel.radii[n1]

        G.add_edge(n1, n2, length=length, direction=direction)# , radius=radius)

    return G

def get_brenches_data(G: nx.Graph, scaling):
    """Return metrics about brenches composing the strcuture encoded in G."""
    
    special_node = [node for node in G.nodes if G.degree[node] != 2]
    brenches = []
    brench_label = 0
    visited_brench = set()
    for sp_node in special_node:
        
        for neighbor in G.neighbors(sp_node):

            edge = tuple(sorted((sp_node, neighbor)))
            if edge in visited_brench:
                continue
            
            length_brench = G.edges[sp_node, neighbor]['length']
            diameters = [G.nodes[sp_node]['radius']]
            visited_brench.add(edge)

            current = neighbor
            past = sp_node
            brench_nodes = [past]
            brench_edges = [edge]

            while G.degree[current] == 2:
                
                next_node = [n for n in G.neighbors(current) if n != past][0]
                edge = tuple(sorted((current, next_node)))
                visited_brench.add(edge)

                length_brench += G.edges[current, next_node]['length']
                diameters.append((G.nodes[current]["radius"]))

                brench_nodes.append(current)
                brench_edges.append(edge)

                past = current
                current = next_node
            
            diameters.append((2*G.nodes[current]["radius"]))
            brench_nodes.append(current)
            tortuosity = length_brench/norm_3d(G.nodes[current]['coord'],G.nodes[sp_node]['coord'])
                
            # if length_brench <= 2/scaling[1]:
            #     brench_type = "noise"
            # elif tortuosity == 1:
            #     brench_type = "noise"
            # elif length_brench <= 10:
            #     brench_type = "sprout"
            # else:
            #     brench_type = "brench"
            brench_type = ""

            brenches.append({
            "Label": brench_label,
            "Endpoints": {sp_node: G.nodes[sp_node]['coord'], current: G.nodes[current]['coord']}, 
            "Length": length_brench, 
            "Mean diameter": np.mean(diameters), 
            "Std diameter": np.std(diameters),
            "Tortuosity": tortuosity,
            "Volume": length_brench * np.mean(diameters),
            "Direction": (np.asarray(G.nodes[current]['coord']) - np.asarray(G.nodes[sp_node]['coord']))
              / norm_3d(np.asarray(G.nodes[current]['coord']), np.asarray(G.nodes[sp_node]['coord'])),
            "Type": brench_type,
            "Note": "",
            "Broken": False,
            "Nodes": [brench_nodes],
            "Edges": [brench_edges]})

            brench_label += 1
    
    return pd.DataFrame(brenches)

def brenches_post_process(G: nx.Graph, brenches: pd.DataFrame, scaling):

    # brenches_updated = brenches[brenches["Type"] != "noise"]
    rm_edges = list(brenches[brenches["Type"] == "noise"]["Edges"])
    
    #Annotate brenches that are smaller than 2voxels:
    brenches.loc[brenches["Length"] < 2/scaling[1], "Type"] = "noise"
    brenches.loc[brenches["Length"] < 2/scaling[1], "Note"] += "<2vox"

    #Remove brenches that are perfectly straight:
    brenches.loc[brenches["Tortuosity"] == 1,"Type"] = "noise"
    brenches.loc[brenches["Tortuosity"] == 1,"Note"] += "straight"

    # #Search node with high number of brenches
    # multi_node = [node for node in G.nodes if G.degree[node] > 3]
    # print("multi_node", multi_node)
    # for m_node in multi_node:
    #     print("m_node:", m_node)
    #     brenches_m = brenches[brenches["Nodes"].apply(lambda x: m_node in x[0])]
    #     print(brenches_m)
    #     brenches_max = brenches_m.loc[brenches_m['Length'].idxmax()]
    #     brenches_rm = brenches_m[brenches_m["Length"] <= 1.5*brenches_max["Mean diameter"]]
    #     print(brenches_rm)
    #     ids_rm = brenches_rm["Label"]
    #     print("ids_rm=", ids_rm, type(ids_rm))
    #     for _,b in brenches.iterrows():
    #         if b["Label"] in ids_rm:
    #             b["Note"] += "multi_n"
    #             b["Type"] = "noise"
    # print(brenches)

    brenches.loc[brenches["Length"] < 10 , "Note"] += "sprout"
    brenches.loc[brenches["Type"] == "", "Type"] = "brench"


    # for _, row in brenches.iterrows():

    #     #High probabability the node contains noise brenches
    #     if any(node in flatten_list_recursive(row["Nodes"], final='int') for node in multi_node):
    #         print(row["Label"])

    brenches_updated = brenches[brenches["Type"] != "noise"]
    rm_edges = list(brenches[brenches["Type"] == "noise"]["Edges"])
    rm_edges = flatten_list_recursive(rm_edges, final='tuple')
    # rm_nodes = [node for node in G.nodes if node not in flatten_list_recursive(brenches_updated["Nodes"], final='int')]
    G_updated = G.copy()
    G_updated.remove_edges_from(rm_edges)
    # G_updated.remove_nodes_from(rm_nodes)

    brenches_straight = brenches[brenches["Note"].str.contains("straight", na=False)]
    rm_edges = brenches[~brenches["Label"].isin(brenches_straight["Label"])]["Edges"]
    rm_edges = flatten_list_recursive(rm_edges, final='tuple')
    G_straight = G.copy()
    G_straight.remove_edges_from(rm_edges)

    brenches_sprout = brenches[brenches["Note"] == "sprout"]
    rm_edges = brenches[~brenches["Label"].isin(brenches_sprout["Label"])]["Edges"]
    rm_edges = flatten_list_recursive(rm_edges, final='tuple')
    G_sprout = G.copy()
    G_sprout.remove_edges_from(rm_edges)

    brenches_vox = brenches[brenches["Note"].str.contains("2vox", na=False)]
    rm_edges = brenches[~brenches["Label"].isin(brenches_vox["Label"])]["Edges"]
    rm_edges = flatten_list_recursive(rm_edges, final='tuple')
    G_vox = G.copy()
    G_vox.remove_edges_from(rm_edges)

    # #Now that some brenches have been removed, need to reconstruct broken branches
    # Alone brenches:
    #   find alone brenches by searching for brench that contain two nodes of degree one
    #   create an edge in between the two closest nodes that includes one node of the alone brench
    #   maybe need to apply slight deformation to avoid perfect straight lines, ie adding a few nodes ?
    # Broken brenches:
    #   search all extremities (ie node of degree 1)
    #   create an edge if the distance between the two extremity is <2*vox
    #   
    # Re-run brench analysis to get updated brenches

    # new_edges = []
    # extremity_node = [node for node in G_updated.nodes if G_updated.degree[node] == 1]

    # for _,row in brenches_updated.iterrows():
    #     keys = list(row["Endpoints"].keys())
    #     if len([True for k in keys if k in extremity_node]) == 2:
    #         #alone_brench = pd.concat([alone_brench, row.to_frame().T], ignore_index=True)
    #         dist_0 = [norm_3d(G_updated.nodes[node]["coord"], G_updated.nodes[keys[0]]["coord"]) for node in G_updated.nodes]
    #         dist_1 = [norm_3d(G_updated.nodes[node]["coord"], G_updated.nodes[keys[1]]["coord"]) for node in G_updated.nodes]
    #         dist_0[dist_0.index(0)] = 1e10
    #         dist_1[dist_1.index(0)] = 1e10
    #         min_0 = np.min(dist_0)
    #         min_1 = np.min(dist_1)
    #         n = min(min_0, min_1)
    #         if n in min_0:
    #             node = min_0.index(n)
    #             k=keys[0]
    #         else:
    #             node = min_1.index(n)
    #             k=keys[1]
    #         print(n)
    #         new_edges.append(tuple(sorted((node, k))))
    #         print(new_edges)
    #         quit()


    return G_updated, brenches_updated, G_straight, G_sprout, G_vox

def get_structure_data(G: nx.Graph, brenches: pd.DataFrame, scaling: tuple):
    """Return metrics about the structure encoded in G."""

    vertices = [G.nodes[n]['coord'] for n in G.nodes]
    bbox_coord = bbox_coordinate(vertices)
    bbox_vol = bbox_volume(bbox_coord, scaling)
    n_brench = brenches.shape[0]
    n_junction = len([d for _,d in G.degree if d> 2])
    n_endpoints = len([d for _,d in G.degree if d == 1])
    n_cycle = len(nx.cycle_basis(G))

    row_global = {
        "Bbox coordinate": [bbox_coord],
        "Total length": brenches["Length"].sum(),
        "Mean Length": brenches["Length"].mean(),
        "Std Length": brenches["Length"].std(ddof=0),
        "Longest Path": None,
        "Longest Path length": None,
        "Lowest node": lowest_node(vertices),
        "Mean diameter": brenches["Mean diameter"].mean(),
        "Std diameter": brenches["Mean diameter"].std(ddof=0),
        "Mean Tortuosity": brenches["Tortuosity"].mean(),
        "Std Tortuosity": brenches["Tortuosity"].std(ddof=0),
        "Total volume": brenches["Volume"].sum(),
        "Bbox volume": bbox_vol,
        "Brench number": n_brench,        
        "Junction number": n_junction,        
        "Loop number": n_cycle,      
        "Endpoint number": n_endpoints        
    }
    if bbox_vol is None:
        density = {
            "Total volume density": None,
            "Brench density": None,
            "Junction density": None,
            "Loop density": None,
            "Endpoint density": None
        }
    else:
        density = {
            "Total volume density": brenches["Volume"].sum()/bbox_vol,
            "Brench density": n_brench/bbox_vol,
            "Junction density": n_junction/bbox_vol,
            "Loop density": n_cycle/bbox_vol,
            "Endpoint density": n_endpoints/bbox_vol
        }

    row_global = row_global |density

    return pd.DataFrame(row_global, index=[0])

def longest_path_tree(G: nx.Graph):
    # Step 1: pick arbitrary node
    start = list(G.nodes)[0]

    # Step 2: find farthest node from it
    lengths = nx.single_source_dijkstra_path_length(G, start, weight='length')
    u = max(lengths, key=lengths.get)

    # Step 3: from u, find farthest node
    lengths, paths = nx.single_source_dijkstra(G, u, weight='length')
    v = max(lengths, key=lengths.get)

    return paths[v], lengths[v]

def bbox_coordinate(vertices):
    """Return bbox coordinate of structure as (z_min, y_min, x_min), (z_max, y_max, x_max)"""
    nodes_coords = pd.DataFrame(vertices, columns=["z", "y", "x"])
    return (tuple(nodes_coords.min()), tuple(nodes_coords.max()))

def lowest_node(vertices):
    """Return the nodes located the lowest on z-axis."""
    nodes_coords = pd.DataFrame(vertices, columns=["z", "y", "x"])
    low_z = nodes_coords.z.min()
    idx_low_z = nodes_coords.index[nodes_coords.z == low_z]
    lowest_node = [tuple(float(e) for e in elt) for elt in nodes_coords.iloc[idx_low_z].values]

    return [tuple(lowest_node)]

def bbox_volume(bbox_coordinate, scaling):
    """Return the volume of the bbox"""

    volume = []
    for i in range(len(bbox_coordinate[0])):
        volume.append(abs(bbox_coordinate[0][i] - bbox_coordinate[1][i]))
    if 0 in volume:
        volume.remove(0)
        if 0 in volume: 
            volume.remove(0)
            return scaling*scaling*volume[0]
        return scaling*volume[0]*volume[1]
    return volume[0]*volume[1]*volume[2]

def get_network_data(endo_vessel_bin):
    """Return the metrics of the network."""

    network_data = None

    return network_data

def colored_zproj(stack_endo, step, zproj_name, display="off"):

    # Normalize intensity values between 0 and 1 (global normalization)
    #-------------------------------------------------------------------
    stack_endo = stack_endo.astype(np.float32) #Convert to float for safe normalization and math operations
    stack_endo = (stack_endo - stack_endo.min()) / (stack_endo.max() - stack_endo.min())

    #Create colors fro z-projection
    #-------------------------------------------------------------------    
    h, y, x = stack_endo.shape # Extract dimensions
    cmap_cont = mcolors.LinearSegmentedColormap.from_list(
        "z_gradient",
        # ["brown", "darkorange", "yellow", "lime", "cyan", "dodgerblue", "magenta", "red"]  # you can tweak this
        ["dodgerblue", "deepskyblue", "cyan", "lime", "yellow", "gold", "darkorange", "red", "magenta"] #seem to give nice rendering
    )
    colors = cmap_cont(np.linspace(0, 1, h))
    # hues = np.linspace(0, 1, h, endpoint=False) # Generate evenly spaced hue values (one per slice). Hue is in [0,1] for HSV color space
    # colors = [mcolors.hsv_to_rgb((hue, 0.85, 1.0)) for hue in hues] # Convert each hue into an RGB color (fixed saturation and value for vivid colors)

    cmap = mcolors.ListedColormap(colors) # Create a discrete colormap using the generated colors

    #Link slices and colors
    #-------------------------------------------------------------------    
    bounds = np.arange(-0.5, h + 0.5, 1) # Define boundaries so that each slice corresponds to one discrete color
    norm = mcolors.BoundaryNorm(bounds, cmap.N)     # Create a normalization object mapping slice indices to colormap entries

    # Maximum Intensity Projection (MIP):
    # For each (x, y) pixel, keep the maximum intensity across z
    mip = np.max(stack_endo, axis=0)

    # Depth map:
    # For each (x, y) pixel, get the index (z-slice) where the max intensity occurs
    depth = np.argmax(stack_endo, axis=0)

    # Normalize depth values to [0,1] so they can be used with the colormap
    depth_norm = depth / (h - 1)

    # Map normalized depth values to RGB colors using the colormap
    depth_color = cmap(depth_norm)

    # Combine color and intensity:
    # - depth_color gives hue (which slice)
    # - mip gives brightness (signal strength)
    # Expand mip to match RGB channels and multiply
    rgb = depth_color[..., :3] * mip[..., None]
    rgb = img_as_ubyte(rgb)

    # Optional display
    plt.imshow(rgb, cmap=cmap, norm=norm)
    plt.axis("off")
    cbar = plt.colorbar(ticks=np.arange(h))
    cbar.set_label("Depth (µm)")
    cbar.set_ticklabels(np.round(np.arange(0, h*step, step), 2))
    plt.clim(-0.5, h - 0.5)
    plt.savefig(zproj_name, dpi=300)
    if display == True:
        plt.show()
    else:
        plt.close()

    # Return the final RGB projection image
    return rgb

def vizualization_structures(skeleton: dict[Skeleton], structure_data: pd.DataFrame, scaling, shape, name):

    structure_id = structure_non_overlapping(structure_data)

    for n in range(len(structure_id)):
        skel_n = {}
        for i in structure_id[n]:
            skel_n[i] = skeleton[i]
        skel_stack = skeletons_to_volume(skel_n, shape, scaling)
        np.save(name + f"_fig{n}.npy", skel_stack)

    # print(len(structure_fig), structure_fig)

def string_2_coord(text):
    
    text = text.replace(")", "")
    text_coord = text.split("(")
    text_coord = [elt.split(",") for elt in text_coord if elt != ""]
    text_coord = [elt.strip() for _ in text_coord for elt in _]
    coord = [float(elt) for elt in text_coord if elt != ""]
    coord = [tuple(coord[0:3]), tuple(coord[3:len(coord)])]
    return coord

def extract_xy_bbox(row):
    (zmin, ymin, xmin), (zmax, ymax, xmax) = row
    return {
        "x_min": xmin,
        "x_max": xmax,
        "y_min": ymin,
        "y_max": ymax
    }

def overlap_2d(a, b):
    return not (
        a['x_max'] <= b['x_min'] or
        a['x_min'] >= b['x_max'] or
        a['y_max'] <= b['y_min'] or
        a['y_min'] >= b['y_max']
    )

def structure_non_overlapping(df):
    # Convert dataframe into list of bbox dicts
    bboxes = [extract_xy_bbox(b) for b in df["Bbox coordinate"]]
    bbox_id = df["Id"].to_list()
    
    groups = []  # list of lists of indices
    groups_id = []

    for idx, bbox in enumerate(bboxes):
        placed = False

        for group, group_id in zip(groups, groups_id):
            # Check if bbox overlaps with any box in the group
            if all(not overlap_2d(bbox, bboxes[g_idx]) for g_idx in group):
                group.append(idx)
                group_id.append(bbox_id[idx])
                placed = True
                break

        if not placed:
            groups.append([idx])
            groups_id.append([bbox_id[idx]])

    return groups_id

