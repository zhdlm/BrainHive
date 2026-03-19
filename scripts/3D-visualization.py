import numpy as np
import napari

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

viewer = napari.Viewer()
endo_stack_contrast  = np.load("endo_stack_contrast.npy")
viewer.add_image(endo_stack_contrast, name='contrast')
endo_vessel_sato_10pxl = np.load("endo_vessel_sato_1-10pxl.npy")
viewer.add_image(endo_vessel_sato_10pxl, name='pxl')
endo_vessel_sato_10um = np.load("endo_vessel_sato_1-10um.npy")
viewer.add_image(endo_vessel_sato_10um, name='1-10um')
endo_vessel_sato_15um = np.load("endo_vessel_sato_1-15um.npy")
viewer.add_image(endo_vessel_sato_15um, name='1-15um')
# endo_vessel_cl_no = np.load("endo_vessel_cl_no.npy")
# viewer.add_image(endo_vessel_cl_no, name='cl_no')
# endo_vessel_cl_no_ero = np.load("endo_vessel_cl_no_ero.npy")
# viewer.add_image(endo_vessel_cl_no_ero, name='cl_no_ero')
# endo_vessel_dil_cl_no = np.load("endo_vessel_dil_cl_no.npy")
# viewer.add_image(endo_vessel_dil_cl_no, name='dil_cl_no')
# endo_vessel_dil_cl_no_ero = np.load("endo_vessel_dil_cl_no_ero.npy")
# viewer.add_image(endo_vessel_dil_cl_no_ero, name='dil_cl_no_ero')

napari.run()
