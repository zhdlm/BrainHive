import numpy as np
import napari
import os

# endo_stack = np.load("endo_stack.npy")
# viewer0 = napari.Viewer()
# viewer0.add_image(endo_stack, name='original')

# endo_stack_contrast = np.load("endo_stack_contrast.npy")
# viewer1 = napari.Viewer()
# viewer1.add_image(endo_stack_contrast, name='contrast')

# endo_stack_iso = np.load("endo_stack_iso.npy")
# viewer2 = napari.Viewer()
# viewer2.add_image(endo_stack_iso, name='sato')

# endo_stack_thresh = np.load("endo_vessel_thresh.npy")
# # viewer3 = napari.Viewer()
# viewer2.add_image(endo_stack_thresh, name='thresh')

# endo_stack_bin = np.load("endo_vessel_bin.npy")
# # viewer4 = napari.Viewer()
# viewer2.add_image(endo_stack_bin, name='bin')

# skeleton_stack = np.load("skeleton_stack.npy")
# # viewer4 = napari.Viewer()
# viewer2.add_image(skeleton_stack, name='skeleton')

path = r"C:\Users\ChimieENS\Documents\Layla\Data\250926_Exp002-E\Plate1\D15-AnalysisNew"
name = "Exp002-E_ShortNI-"

viewer = napari.Viewer()

sato = np.load(os.path.join(path, name + "_sato.npy"))
bin = np.load(os.path.join(path, name + "_bin.npy"))
# bin2 = np.load(os.path.join(path, name + "_bin2.npy"))
# skel = np.load(os.path.join(path, name + "_skeleton.npy"))
# skelopti = np.load(os.path.join(path, name + "_skeleton_opti.npy"))
# skelfin = np.load(os.path.join(path, name + "_skeleton_final.npy"))

viewer.add_image(sato, name="sato")
viewer.add_image(bin, name="bin")
# viewer.add_image(bin2, name="bin2")
# viewer.add_image(skel, name="skel")
# viewer.add_image(skelopti, name="skelopti")
# viewer.add_image(skelfin, name="skelfin")

napari.run()
