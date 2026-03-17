import os
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu, threshold_yen, rank, threshold_local, threshold_multiotsu
from skimage.morphology import binary_erosion
from skimage.morphology import disk
import czifile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd
from pprint import pprint

VERSION = "V0"
day_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal"
out_path = r"C:\Users\ChimieENS\Documents\Layla\Data\251117_Exp004-E\D14-confocal-Analysis"
filename = "Exp004-E_F5_Seqlong_x10_GFP-BF-Cy3-DAPI_16bit-1024-n4-s7_D14_2025_12_01__16_48_47.czi"
fluo_endo = "GFP"
fluo_chans = ["GFP","CY3","BF","DAPI"]
pxl2um = 1.384
um2pxl = 1/pxl2um
radius_blur = 5 #in µm
radius_bkgd = 100
radius_local = 50
radius_erosion = 5
z_start = 0
z_end = -1 #set to -1 if you want to go until the end of the stack
rm_noise = False

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

#Open image
img_all = czifile.imread(day_path + path_delimiter + filename)
z_stack = img_all.shape[4]
dim = (img_all.shape[5], img_all.shape[6])

idx_fluo_endo = fluo_chans.index(fluo_endo)
results = []

if z_end == -1: z_end = z_stack

for i, z in enumerate(range(z_start, z_end)):
    
    print(z)

    #Open image
    img = img_all[0,0,0,idx_fluo_endo,z,0:dim[0],0:dim[1], 0]

    #Blur image
    radius = radius_blur*um2pxl
    img_blur = gaussian_filter(img, radius)

    #bkgd
    if rm_noise == True:
        img_bkgd = gaussian_filter(img, radius_bkgd)
        n = round(dim[0]/2)
        bkgd_means = [
            img_bkgd[0:n, 0:n].flatten().mean(),
            img_bkgd[0:n, n:-1].flatten().mean(),
            img_bkgd[n:-1, 0:n].flatten().mean(),
            img_bkgd[n:-1, n:-1].flatten().mean()
        ]
        print(np.mean(bkgd_means), np.std(bkgd_means), 100*np.std(bkgd_means)/np.mean(bkgd_means))
        img_no_bkgd = img_blur - img_bkgd
    else:
        img_no_bkgd = img_blur


    #Determine threshold
    footprint = disk(radius_local)
    #local_otsu = rank.otsu(img_blur, footprint)
    global_3otsu_no_bkgd = threshold_multiotsu(img_no_bkgd, 3)
    global_otsu_no_bkgd = threshold_otsu(img_no_bkgd)
    thresh_value = (global_3otsu_no_bkgd[0] + global_otsu_no_bkgd)/2
    # img_bin_otsu = img_no
    nbins = np.power(2, 16) - 1
    print(global_3otsu_no_bkgd, global_otsu_no_bkgd, thresh_value)

    #Apply threshold
    img_thresh = (img_no_bkgd >= thresh_value)

    #Erode image
    img_bin = binary_erosion(img_thresh, disk(round(radius_erosion*um2pxl/2)))

    #Store mask
    img_bin_8bit = np.ndarray(dim, dtype='uint8')
    img_bin_8bit[img_bin == True] = 255
    img_bin_8bit[img_bin == False] = 0

    #overlay
    overlay = np.ndarray((dim[0], dim[1], 3), dtype='uint8')
    img_8bit = img_as_ubyte(img)
    overlay[:,:,0] = img_8bit
    overlay[:,:,1] = img_8bit + 0.2*img_bin_8bit
    overlay[:,:,2] = img_8bit

    # fig, ax = plt.subplots(2,3, num=f"{z}")
    # ax[0, 0].imshow(img)
    # ax[0, 1].imshow(img_no_bkgd)
    # ax[0, 2].imshow(img_thresh)
    # ax[1, 0].imshow(img_bin)
    # ax[1, 1].imshow(overlay)
    # ax[1, 2].hist(img_no_bkgd.flatten(), 255)
    # ax[1, 2].scatter(global_3otsu_no_bkgd, [0, 0], color='r')
    # ax[1, 2].scatter(global_otsu_no_bkgd, 0, color='g')

    #Store outputs
    tmp = img.copy()
    tmp[~img_bin] = 0
    results.append({
        "Z": z,
        "Area occupied": np.count_nonzero(img_thresh),
        "% of area occupied": 100*np.count_nonzero(img_thresh) / (dim[0]*dim[1]),
        "Raw Mean Intensity": img.flatten().mean(),
        "Raw Std Intensity": img.flatten().std(),
        "Mask Mean Intensity": tmp.flatten().mean(),
        "Mask Std Intensity": tmp.flatten().std(),
        "Image Name": filename
    })


#Display maximum Z-projection for visualization 
stack = np.squeeze(img_all)
stack = stack[0, z_start:z_end+1, :, :]
stack = stack.astype(np.float32)
stack = (stack - stack.min()) / (stack.max() - stack.min())
h, y, x = stack.shape
hues = np.linspace(0, 1, h, endpoint=False)
colors = [mcolors.hsv_to_rgb((hue, 0.85, 1.0)) for hue in hues]
cmap = mcolors.ListedColormap(colors)
bounds = np.arange(-0.5, h + 0.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)
mip = np.max(stack, axis=0)
depth = np.argmax(stack, axis=0)
depth_norm = depth / (h-1)
depth_color = cmap(depth_norm)
rgb = depth_color[..., :3] * mip[..., None]
plt.imshow(rgb, cmap=cmap, norm=norm)
plt.axis("off")
plt.colorbar(ticks=np.arange(h))
plt.clim(-0.5, h - 0.5)
plt.show()

#Save outputs
results = pd.DataFrame(results)
filename.replace('.czi', '.csv')
results.to_csv(out_path + path_delimiter + filename)
filename.replace('.csv', '.jpg')
imsave(out_path + path_delimiter + filename, rgb)
pprint(results)







