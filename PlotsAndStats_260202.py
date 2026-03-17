import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns
from scipy.stats import median_abs_deviation, shapiro, levene, kruskal, mannwhitneyu, false_discovery_control
from typing import List
from enum import Enum
import os
import numpy as np
import itertools
import warnings
from pingouin import compute_effsize

from custom_functions import read_usr_input, open_results, add_group_counts

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

def simple_warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"
warnings.formatwarning = simple_warning_format

class PlotType(Enum):
    VIOLIN = "violin"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    SCATTER_MEAN = "scatter_mean"
    SCATTER_MEDIAN = "scatter_median"
    LINE_MEAN = "line_mean"
    LINE_MEDIAN = "line_median"
    SWARMPLOT = "swarmplot"

########################################
# Stats
########################################

def stat_decision_tree(data: pd.DataFrame, variable: str, group_columns: List[str]):
    
    #Verify required headers are present
    for elt in (*group_columns, variable):
        check_header(elt, data.columns.values.tolist())
    if None in group_columns:
        group_columns.remove(None)
    print(group_columns)
    
    #Compute basic stats per group
    data_grouped = data.groupby(by=group_columns)[variable]
    stat_group = data_grouped.describe()
    stat_group["IQR"] = stat_group["75%"] - stat_group["25%"] #adding IQR to stats
    group_names = list(stat_group.index)
    data_grouped_list = []
    for group in group_names:
        data_grouped_list.append(list(data_grouped.get_group(group)))

    #Decide whether to perform parametric or non-parametric test based on group size
    if any((stat_group["count"].values < 5)): #Warning + tag
        warnings.warn("At least one group have less than 5 objets. Consider removing it from plot & stats.", UserWarning)
        decision = 'kruskal'

    elif any(stat_group["count"].values < 10): #Non-parametric
        print("At least one group contains less than 10 objects, therefore, non-parametric tests is performed.")
        decision = 'kruskal'

    elif all(stat_group["count"].values >= 20) : #Normality + variance then Parametric if pass else nonparametric

        print("All groups contain 20 or more objects, therefore, Normality and Variance equality are tested.")

        #Perform normality test using Shapiro method
        shapiro_test = {}
        for group, group_data in zip(group_names, data_grouped_list):
            shapiro_test[group] = shapiro(group_data)

        #Perform Variance equality test
        levene_test = levene(*data_grouped_list)

        if all(shapiro_test.values) > 0.05 and levene_test > 0.05:
            print("Normal distribution: No.\nMedian Variance equality: No.")
            decision = 'kruskal'
        else:
            print("Data fits to null hypothesis or to median variance equality, therefore, Parametric Global Test One-way ANOVA is used")
            decision = '1WANOVA'
        
    else: #Non parametric
        print("All groups have in between 11 and 19 objects, therefore  Non-Parametric Global Test Kruskal Wallis is used")
        decision = 'kruskal'

    #Global test
    if decision == 'kruskal': # Non parametric Kruskal Wallis test, then pair Mann-Whitney & Effect Size
        kruskal_test = kruskal(*data_grouped_list)

        if kruskal_test.pvalue < 0.05:
            print(f"Kruskal-Wallis H Test: {kruskal_test.pvalue} --> Chi squared distribution.")

            #Pair wise test Mann-Whitney U & Cliff's Delta
            mannwhitneyu_stat = {}
            pvals = []
            cliffdelta = []
            for comb in itertools.combinations(range(len(group_names)), 2):
                print(comb)
                comb_name = str(group_names[comb[0]]) + " vs " + str(group_names[comb[1]])
                mannwhitneyu_stat[comb_name] = mannwhitneyu(data_grouped_list[comb[0]], data_grouped_list[comb[1]])
                pvals.append(mannwhitneyu_stat[comb_name].pvalue)
                #cliffdelta.append(compute_effsize(data_grouped_list[comb[0]], data_grouped_list[comb[1]], eftype=))
            
            #Adjust p-values using the Benjamini-Hochberg Theorem
            pvals_adjusted = false_discovery_control(pvals)

            #Prints
            for i,elt in enumerate(mannwhitneyu_stat):
                print(f"{elt}: pvalue={mannwhitneyu_stat[elt].pvalue}, pvalue_bh_adjusted={pvals_adjusted[i]}")#, cliif={cliffdelta[i]}")


        else:
            print(f"Kruskal-Wallis H Test: {kruskal_test.pvalue} --> Not a Chi squared distribution, therefore no evidence of difference between groups.")
            print()
    
    elif  decision == '1WANOVA': # Parametric 1 Way ANOVA
        pass


#Read user inputs
graph_user_inputs = read_usr_input("graph_inputs.txt")
print(graph_user_inputs)
if len(graph_user_inputs["Graph"]) == 0:
    print("No graph requested.")

#Open, Combine & Save results
data = open_results(graph_user_inputs, path_delimiter)
print("Data loaded:\n", data.columns.values)
data.to_csv(graph_user_inputs["Outpath"])

#Add Day number
tmp = data["Day"].replace("D", "", regex=True)
data["Day Number"] = tmp
data["Day Number"] = data["Day Number"].astype(int) #will create issue if - are present !!!!!!!
del(tmp)

#Add wrinkling index if don't exist
if "wrinkling_index" not in data.columns.values:
    data["equivalent_diameter_perimeter"] = data["perimeter"]/np.pi
    data["wrinkling_index"] = data["equivalent_diameter_perimeter"]/data["equivalent_diameter_area"]

#Convert to um if not done, else comment the 4 lines below
to_convert_1 = ["equivalent_diameter_area", "equivalent_diameter_perimeter"]
for elt in to_convert_1:
    data[elt] = data[elt]*0.9
data["area_filled"] = data["area_filled"]*0.9*0.9



channels = ['dapi', 'gfp', 'cy3']
dyes = ['DAPI', 'Calcein', 'PI']
colors = ['b', 'g', 'r']
noise = [478*1.11, 98*1.05, 161*1.3]
conditions = []

########################################
# FLUO
########################################

#2D fluo plot: violin plot for each signal
def violinplot_fluoI(channels, dyes, noise, data):
    for i in range(len(channels)):
        fig, ax = plt.subplots(num=f"{dyes[i]} on {channels[i]} fluorescent channel (Zeiss Confocal Microscope) at D2")
        sub_data = data[data["Dye"] == dyes[i]]
        sns.violinplot(data=sub_data, x="Dye Concentration", y=f"{channels[i]} Mean Intensity", hue="Well State")
        add_group_counts(ax, data=sub_data, x="Dye Concentration", hue="Well State")
        x_lim = ax.get_xlim()
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.plot(x_lim, [noise[i], noise[i]], c=colors[i])
        ax.set_title(f"{dyes[i]} on {channels[i]} fluorescent channel (Zeiss Confocal Microscope) at D2", fontsize=25)
        ax.set_ylabel(f"{channels[i]} Mean fluorescent intensity per detected object", fontsize=20)
        ax.set_xlabel(f"{dyes[i]} concentration", fontsize=20)
    plt.show()

#2D fluo plot: intensity as function concentration and size
def scatterplot_yFluoI_xArea_szDyeConcentration(channels, dyes, noise, data):
    for i in range(len(channels)):
        fig, ax = plt.subplots(num=f"{dyes[i]} on {channels[i]} fluorescent channel (Zeiss Incubator Microscope) at D1")
        sub_data = data[data["Dye"] == dyes[i]]
        sns.scatterplot(data=sub_data, x="Areas", y=f"{channels[i]} Mean Intensity", hue="Well State", size="Dye Concentration")
        # add_group_counts(ax, data=sub_data, x="Dye Concentration", hue="Well State")
        x_lim = ax.get_xlim()
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.plot(x_lim, [noise[i], noise[i]], c=colors[i])
        ax.set_title(f"{dyes[i]} on {channels[i]} fluorescent channel (Zeiss Incubator Microscope) at D1", fontsize=25)
        ax.set_ylabel(f"{channels[i]} Mean fluorescent intensity per detected object", fontsize=20)
        ax.set_xlabel(f"{dyes[i]} concentration", fontsize=20)
    plt.show()

########################################
# Organoid
########################################

def plot_design(ax):
    ax.set_fontsize()

def check_header(variable: str, headers: List[str]):

    if variable != None:
        if variable not in headers:     
            raise ValueError(f"Variable {variable} do not exist.")

def set_axis_label(column_name):

    name = column_name

    match name:
        case "equivalent_diameter_area" | "equivalent_diameter_perimeter":
            name += " (µm)"
        case "area_filled":
            name += " (µm²)"
        case "centroidX", "centroidY" | "centroid_localX" | "centroid_localY":
            name += " (pxl)"
        case "intensity_mean", "intensity_std" | "intensity_median" | "intensity_mad":
            name += " (A.U.)"
        case "Day":
            name = "Culture time (day)"

    name = name.replace("_", " ")

    return name

def plots(data: pd.DataFrame, x: List[str], y: List[str], conditions: List[str], plot_type: List[PlotType], z: str=None, savefig: bool=False, stats: bool=False):
    """
    Plots of y variable over time.

    Parameters
    ----------
    data: pandas DataFrame.
        Containing the "Day" column (used for the X-axis) and the column listed in "y" that will be used for the Y-axis.
    conditions: list of string.
        The list 
    y: list of PlotType.
        Containing the variable to plot on the Y-axis. The variable need to exist as column headers in the data pd.DataFrame
    plot_type: list of string.
        Containing the name of the desired plots
    """
    
    #Verify the required headers are present in data
    for elt in (*x, *y, *conditions, z):
        check_header(elt, data.columns.values.tolist())

    #Remove organoid that have been manually discarded
    data = data[data["Manually Discarded"] == False]
    
    for i_plot in range(len(plot_type)):

        #Create the figure and set the design
        condition_name = data[conditions[i_plot]].unique()
        if z != None:
            z_name = data[z].unique()
        else:
            z_name="None"
        ttl = f"{plot_type[i_plot]}_x{x[i_plot]}_y{y[i_plot]}_z{",".join(z_name)}_c{",".join(condition_name)}"
        fig, ax = plt.subplots(num=ttl)
        ax.set_xlabel(set_axis_label(x[i_plot]), fontsize=15)
        ax.set_ylabel(set_axis_label(y[i_plot]), fontsize=15)
        ax.set_title(ttl.replace("_", " "))
        for ticks in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            ticks.set_fontsize(15)

        #Plots
        match plot_type[i_plot]:
            case "violin":
                sns.violinplot(data=data, x=x[i_plot], y=y[i_plot], hue=conditions[i_plot])
                add_group_counts(ax, data=data, x="Day", hue=conditions[i_plot])
                if z != None:
                    sns.swarmplot(data=data, x=x[i_plot], y=y[i_plot], hue=z)

            case "boxplot":
                if z != None:
                    sns.boxplot(data=data, x=x[i_plot], y=y[i_plot], hue=conditions[i_plot], fill=False)
                    sns.swarmplot(data=data, x=x[i_plot], y=y[i_plot], hue=z)
                else:
                    sns.boxplot(data=data, x=x[i_plot], y=y[i_plot], hue=conditions[i_plot], fill=True)
                add_group_counts(ax, data=data, x=x[i_plot], hue=conditions[i_plot])

            case "scatter":
                if x[i_plot] == "Day":
                    tmp = data[x[i_plot]].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                    data["Day Number"] = tmp
                    data["Day Number"] = data["Day Number"].astype(int) 
                    x[i_plot] = "Day Number"
                sns.scatterplot(data=data, x=x[i_plot], y=y[i_plot], hue=conditions[i_plot], size=z)

            case "scatter_mean":
                if x[i_plot] == "Day":
                    tmp = data[x[i_plot]].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                    data["Day Number"] = tmp
                    data["Day Number"] = data["Day Number"].astype(int) 
                    x[i_plot] = "Day Number"
                if z != None:
                    means_stds = data.groupby(
                        [conditions[i_plot], x[i_plot]], as_index=False).agg(y_mean=(y[i_plot], "mean"), y_std=(y[i_plot], "std"), z_mean=(z, "mean"))
                    print(means_stds)
                else:
                    means_stds = data.groupby([conditions[i_plot], x[i_plot]], as_index=False).agg(y_mean=(y[i_plot], "mean"), y_std=(y[i_plot], "std"))
                sns.scatterplot(data=means_stds, x=x[i_plot], y="y_mean", hue=conditions[i_plot], size="z_mean")
                plt.errorbar(means_stds[x[i_plot]], means_stds["y_mean"], yerr=means_stds["y_std"],
                    fmt="none",  # don't draw additional markers
                    ecolor="gray",
                    capsize=4
                )

            case "scatter_median":
                if x[i_plot] == "Day":
                    tmp = data[x[i_plot]].replace("D", "", regex=True) #will create issue if - are present !!!!!!!
                    data["Day Number"] = tmp
                    data["Day Number"] = data["Day Number"].astype(int) 
                    x[i_plot] = "Day Number"
                medians_stds = (
                    data.groupby([conditions[i_plot], x[i_plot]], as_index=False)
                    .agg(y_median=(y[i_plot], "median"), y_mad=(y[i_plot], lambda x: median_abs_deviation(x, scale='normal'))))
                sns.scatterplot(data=medians_stds, x=x[i_plot], y="y_median", hue=conditions[i_plot], size=z)
                plt.errorbar(medians_stds[x[i_plot]], medians_stds["y_median"], yerr=medians_stds["y_mad"],
                    fmt="none",  # don't draw additional markers
                    ecolor="gray",
                    capsize=4
                )
            case "line_mean":
                if x[i_plot] == "Day":
                    x[i_plot] = "Day Number"
                sns.lineplot(data=data, x=x[i_plot], y=y[i_plot], )
            case "line_median":
                pass

            case "swarmplot":
                if x[i_plot] == "Day":
                    x[i_plot] = "Day Number"
                sns.swarmplot(data=data, x=x[i_plot], y=y[i_plot], hue=conditions[i_plot])

        #Stats
        stat_decision_tree(data, y[i_plot], [x[i_plot], z, *conditions])
                

    plt.show()
        

# plots(data, ["Day"], ["eccentricity"], ["Condition"], ["boxplot"], z="Substrate")

# fig, ax1 = plt.subplots()

# sns.boxplot(data=data, x="Day Number", y="equivalent_diameter_area", hue="Condition", ax=ax1)
# sns.swarmplot(data=data, x="Day Number", y="equivalent_diameter_area", hue="Substrate", ax=ax1)
# ax2 = ax1.twinx()
# sns.boxplot(data=data, x="Day Number", y="wrinkling_index", hue="Condition", ax=ax2, color="r")
# sns.swarmplot(data=data, x="Day Number", y="wrinkling_index", hue="Substrate", ax=ax2)
# plt.show()



#stat_decision_tree(data, "equivalent_diameter_area", ["Day", "Condition"])
plots(data, ["Day"], ["eccentricity"], ["Condition"], ["boxplot"])
