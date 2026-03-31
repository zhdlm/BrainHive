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

VERSION = "V2"
VESSEL_DIAM = 10 #in µm
CELL_RADIUS = 4 #in µm
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"{os.path.basename(__file__)} | Version={VERSION}")
logger.info("\n=================================\n      MICRO-VESSEL ANALYSIS      \n=================================")

# 0. USER INPUTS
#---------------
usr = vm.read_usr_inputs("User_inputs\\Exp006-E.csv")

for f in range(len(usr["filename"])):

    if usr["condname"][f] != "Seq1-x20-DeltaZ2um":
        continue

    # 1. LOADING
    #-----------
    endo_stack, scaling = vm.load_fluo_stack(usr, f)
    # endo_stack = imread(os.path.join(usr["out_path"], usr["filename"][f]))
    # scaling = (1, 0.3459440920682663, 0.3459440920682663)

    # # 2. PRE-PROCESSING: depends on anisotropy
    # #-----------------------------------------
    # endo_pre_process = vm.pre_processing(usr, f, endo_stack, scaling, VESSEL_DIAM)
    # # endo_pre_process = np.load(os.path.join(usr["out_path"], "Demovascu_sato.npy"))

    # # 3. THRESHOLD
    # #-------------
    # endo_thresholded = vm.threshold(endo_pre_process)

    # # 4. MORPHOLOGY FILTERING
    # #-------------------------
    # endo_bin = vm.morpho_filter(usr, f, endo_thresholded, scaling, CELL_RADIUS)

    # 5. SKELETONIZATION
    #-------------------
    #Optional: re-run only skeleton and comment step 2-4
    name= usr["exp_name"] + usr["condname"][f] + "_bin.npy"
    endo_bin = np.load(os.path.join(usr["out_path"], name))
    #skeletonization
    skeleton_kimi = vm.skeletonization(usr, f, endo_bin, (scaling[1], scaling[1], scaling[1]))

    # 6. METRICS EXTRACTION & DATA SAVE
    #----------------------------------
    brenches_data, structures_data = vm.extract_save_metrics(usr, f, endo_bin, skeleton_kimi, (scaling[1], scaling[1], scaling[1]))

    # ##########################################################
    # # 6. SAVING DATA
    # ##########################################################

    # logger.info("\n---------------------------------\n     Saving Data     \n---------------------------------")
    # logger.info(f"Saving Data | Saving path: {usr["out_path"]}")

    # brench_name = usr["exp_name"] + usr["condname"][f] + "_Brenches_Results.csv"
    # brenches_data.to_csv(os.path.join(usr["out_path"], brench_name), index=False)
    # logger.info(f"Saved brenches data as {brench_name}")

    # structure_name = usr["exp_name"] + usr["condname"][f] + "Structure_Results.csv"
    # structures_data.to_csv(os.path.join(usr["out_path"], structure_name), index=False)
    # logger.info(f"Saved strcuture data as {structure_name}'")

    # updated_skeleton_stack = vm.skeletons_to_volume(skeleton_kimi, endo_vessel_bin.shape, scaling[1])
    # np.save(os.path.join(usr["out_path"], (usr["exp_name"] + usr["condname"][f] + "_skeleton_updated.npy")), updated_skeleton_stack)


    # ##########################################################
    # # 6. VIZUALIZATION
    # ##########################################################

    # # Z-projection with colors
    # endo_stack_opti_contrast = equalize_adapthist(endo_stack)
    # zproj_name = os.path.join(usr["out_path"], usr["exp_name"] + usr["condname"][f] + "contrat_colored_Zproj.jpeg")
    # zproj = vm.colored_zproj(endo_stack_opti_contrast, scaling[0], zproj_name, display=False)
    # zproj_name = os.path.join(usr["out_path"], usr["exp_name"] + usr["condname"][f] + "colored_Zproj.jpeg")
    # zproj = vm.colored_zproj(endo_stack, scaling[0], zproj_name, display=False)