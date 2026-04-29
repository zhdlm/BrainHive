import _bootstrap
import sys
print(sys.executable)
from os.path import basename
from src.core import Exp, Substrate, Day, Hive
from src.options import ImageType, ChannelNames, OrganoidProperties, PlotType, SCALE_RATIO, VERSION, PATH_DELIMITER
import utils.usefull_functions as uf
from utils.logging_setup import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"{basename(__file__)} | Version={VERSION} | scale_ratio={SCALE_RATIO} | path_delimiter={PATH_DELIMITER}")

#1. Read user inputs
user_inputs = uf.read_graph_inputs("User_Inputs/graph_inputs.json")

#2. Open, Combine & Save results
data = uf.open_results(user_inputs, "*Mask_Results.csv", save=False)

conds = ["E-", "E+", "EB-", "EB+", "NI-", "NI+", "D-+-", "D-++", "D--", "D-+", "D+-", "D++"]
data = data[data["Condition"].isin(conds)]
print(data)

#3. Plot graphic
uf.plots(data, "Condition", "Volume Density", "Depth", PlotType.BARPLOT_STACK, savefig=user_inputs["Outpath"], z="Experiment")
# uf.plots(data, "Condition", "Mean Length", "Experiment", PlotType.BARPLOT, savefig=user_inputs["Outpath"])
# uf.plots(data, "Condition", "Mean diameter", "Experiment", PlotType.BARPLOT, savefig=user_inputs["Outpath"])
# uf.plots(data, "Condition", "Total volume", "Experiment", PlotType.BARPLOT_SUM, savefig=user_inputs["Outpath"])

# #4. Perform statistical tests
# stats, title = uf.conservative_stat_decision_tree(data, "area_filled", ["Condition", "Day", "Experiment"], 'continuous', 'difference', day_dependance='match')
# uf.pretty_print(stats, save=True, path=user_inputs["Outpath"], ttl=title)