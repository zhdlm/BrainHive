import _bootstrap
import os
import src.vessel_methods as vm
import utils.usefull_functions as uf
from utils.logging_setup import setup_logging
import logging
from skimage.io import imread, imsave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

VERSION = "V2"
VESSEL_DIAM = 10 #in µm
CELL_RADIUS = 4 #in µm
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"{os.path.basename(__file__)} | Version={VERSION}")
logger.info("\n=================================\n      MICRO-VESSEL ANALYSIS      \n=================================")

# 0. USER INPUTS
#---------------
usr = vm.read_usr_inputs("User_inputs\\Exp001-E_E-.csv")

for f in range(len(usr["filename"])):

    # 1. LOADING
    #-----------
    endo_stack, scaling = vm.load_fluo_stack(usr, f)
    # endo_stack = imread(os.path.join(usr["out_path"], usr["filename"][f]))
    # scaling = (1, 0.3459440920682663, 0.3459440920682663)

    # 2. PRE-PROCESSING: depends on anisotropy
    #-----------------------------------------
    endo_pre_process = vm.pre_processing(usr, f, endo_stack, scaling, VESSEL_DIAM)

    # 3. THRESHOLD
    #-------------
    #Optional: re-run only skeleton and comment step 2-4
    # name= usr["exp_name"] + usr["condname"][f] + "_sato.npy"
    # #endo_pre_process = vm.load_lossless_compressed_img(os.path.join(usr["out_path"]), name)
    # endo_pre_process = np.load(os.path.join(usr["out_path"], name))
    # print(endo_pre_process.shape)
    endo_thresholded = vm.threshold(endo_pre_process)

    # 4. MORPHOLOGY FILTERING
    #-------------------------
    endo_bin = vm.morpho_filter(usr, f, endo_thresholded, scaling, CELL_RADIUS)

    # 5. SKELETONIZATION
    #-------------------
    # #Optional: re-run only skeleton and comment step 2-4
    # name= usr["exp_name"] + usr["condname"][f] + "_bin.npy"
    # endo_bin = np.load(os.path.join(usr["out_path"], name))
    #skeletonization
    skeleton_kimi = vm.skeletonization(usr, f, endo_bin, (scaling[1], scaling[1], scaling[1]))

    # 6. METRICS EXTRACTION & DATA SAVE
    #----------------------------------
    brenches_data, structures_data = vm.extract_save_metrics(usr, f, endo_bin, skeleton_kimi, (scaling[1], scaling[1], scaling[1]))

    #vm.vizualization_structures(skeleton_kimi, structures_data, (scaling[1], scaling[1], scaling[1]), endo_bin.shape, usr["condname"][f])


    # 7. VIZUALIZATION
    #----------------------------------
    endo_stack_opti_contrast = equalize_adapthist(endo_stack)
    zproj_name = os.path.join(usr["out_path"], usr["exp_name"] + usr["condname"][f] + "contrat_colored_Zproj.jpeg")
    zproj = vm.colored_zproj(endo_stack_opti_contrast, scaling[0], zproj_name, display=False)
    # zproj_name = os.path.join(usr["out_path"], usr["exp_name"] + usr["condname"][f] + "colored_Zproj.jpeg")
    # zproj = vm.colored_zproj(endo_stack, scaling[0], zproj_name, display=False)