import _bootstrap
from os.path import basename
from src.core import Exp, Substrate, Day, Hive
from src.options import ImageType, ChannelNames, SCALE_RATIO, VERSION, PATH_DELIMITER
from utils.logging_setup import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"{basename(__file__)} | Version={VERSION} | scale_ratio={SCALE_RATIO} | path_delimiter={PATH_DELIMITER}")

#1. Load input file: aquisition parameters
exp = Exp._from_csv("Exp01.csv")
exp.display_hierarchy(compact=True) #If you want to display

#2. Set the subset on whiwh you want to work (comment the undesired line) 
hierarchy = None #in case you want to work on all data set
hierarchy = { #in case you have a subset only to work on	
    "5X": ["1"]
}

#3. Crop & Refine Crop to have only one hive per image
exp.crop(hierarchy=hierarchy, save_montage=True)
exp.optimized_crop(hierarchy=hierarchy)

#4. Compute threshold & generate organoid mask (takes about 30sec/image)
exp.threshold(hierarchy=hierarchy, verbose='off', img_type=ImageType.OPTIMIZED_SHADOW)
exp.get_organoid_mask(hierarchy=hierarchy, img_type=ImageType.OPTIMIZED_SHADOW)
exp.create_montage(ImageType.OVERLAY, hierarchy=hierarchy)

#Optionnal: you may look at the Overlay Montages and optionnally manually modify the organoid mask
#on ImageJ (cf README for more info and procedure.)

#5. Save organoid data & manually discard bad segmentation
exp.get_save_organoid_data(hierarchy=hierarchy, img_type=ImageType.OPTIMIZED_SHADOW)
exp.manual_discard(hierarchy=hierarchy)
