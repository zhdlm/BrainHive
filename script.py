import matplotlib.pyplot as plt
from core20260305 import Exp, Substrate, Day, Hive, ImageType, ChannelNames, HexagonMath, hexagon_outline_ndarray, display_montage_organoids_dynamic

# exps = ["\\InputFiles_csv\\Exp001-B.csv", "\\InputFiles_csv\\Exp002-B.csv", "\\InputFiles_csv\\Exp003-B.csv",
#         "\\InputFiles_csv\\Exp004-B.csv", "\\InputFiles_csv\\Exp005-B.csv", "\\InputFiles_csv\\Exp006-B.csv", 
#         "\\InputFiles_csv\\Exp007-B.csv", "\\InputFiles_csv\\Exp008-B.csv"]

#1. Load experiment: 
exp1 = Exp._from_csv("InputFiles_csv\\Demo.csv")

#2. Optionnal: Display the loaded data
exp1.display_hierarchy(compact=True)

# #3. Optionnal: define the subset on which you want to perform the following steps
# hierarchy = None
hierarchy = {
    "Old-2": ["D6"]
}

# #4. Crop the pictures:
# exp1.crop(save_montage=True, hierarchy=hierarchy)

# #5. Optimized crop of the pictures:
# exp1.optimized_crop(save_montage=False, hierarchy=hierarchy)

# #6. Determine threshold for organoid & create mask: This step is quite long !
# exp1.threshold(display='off', verbose='off', hierarchy=hierarchy)
# exp1.get_organoid_mask(hierarchy=hierarchy, channel=None)

# #8. Get & Save Organoid Data
# exp1.get_save_organoid_data(hierarchy=hierarchy)

#9. Perform manual discard & Update organoid data
# exp1.create_montage(ImageType.CROP, hierarchy=hierarchy)
# exp1.create_montage(ImageType.OVERLAY, hierarchy=hierarchy)
# exp1.manual_discard(hierarchy=hierarchy)
hierarchy = {
    "Exp009-B": {
        "Old-1": [5,9,13],
        "Old-2": [10,14,8]
    }

}
exp1.display_dynamic(hierarchy, ["D1", "D4", "D6"], (0,0,0,ChannelNames.BF.name), ImageType.OPTIMIZED_SHADOW)