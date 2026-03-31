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

path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal-analysis"

viewer = napari.Viewer()

iso= np.load(os.path.join(path,"Exp004-E_D-+CHIR-_iso.npy"))
sato = np.load(os.path.join(path,"Exp004-E_D-+CHIR-_sato.npy"))
bin = np.load(os.path.join(path,"Exp004-E_D-+CHIR-_bin.npy"))
viewer.add_image(sato, name="sato")
viewer.add_image(bin, name="bin")

for s in [3, 3.5, 5]:
    for c in [0,0.5, 1]:
        for e in [8]:
            params = f"s{s}_c{c}_e{e}"
            name = f"Exp004-E_D-+CHIR-_skeleton_s{s}_c{c}_e{e}.npy"
            skel = np.load(os.path.join(path, name))
            viewer.add_image(skel, name=params)

napari.run()
