import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns
from scipy.stats import median_abs_deviation, shapiro, levene, kruskal, mannwhitneyu, false_discovery_control, fisher_exact, chisquare, ttest_rel, ttest_ind, wilcoxon, bartlett, f_oneway , tukey_hsd, friedmanchisquare, zscore, skew
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from typing import List
from enum import Enum
import os
import numpy as np
import itertools
import warnings
from core20260305 import OrganoidProperties, unique_list
from custom_functions import read_usr_input, open_results, add_group_counts
from pprint import pprint

os_name = os.name
if os_name == 'posix':
    path_delimiter = '/'
elif os_name == 'nt':
    path_delimiter = '\\'

def simple_warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"
warnings.formatwarning = simple_warning_format

def pretty_print(data: dict):
    for key, value in data.items():
        print(f"\n=== {key} ===")
        
        if key == "stats":
            df = pd.DataFrame(value)
            print(df.to_string(index=False))
        elif key in ["Dependant", "Independant"]:
            for group, group_value in value.items():
                print(f"\n---Test on groups: {group}---")
                for sub_key, sub_value in group_value.items():
                    print(f"\t--{sub_key}--")
                    if sub_value is None:
                        pprint(None)
                    elif isinstance(sub_value, list):
                        df = pd.DataFrame(sub_value)
                        print(df.to_string(index=False))
                    else:
                        pprint(sub_value)
        else:
            pprint(value)
        print("============================================================")

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

def structure_data(data_grouped_as_index, group, variable, request):

    #print(data_grouped_as_index, group, variable, request)

    if request == 'array_like':
        data_group = []
        for g in group:
            if g in data_grouped_as_index.groups.keys():
                values = data_grouped_as_index.get_group(g)[variable]
                order = data_grouped_as_index.get_group(g)["Hive_number"]
                sorted_values = [val for _, val in sorted(zip(order, values))]
                data_group.append(sorted_values)

    elif request == 'dataframe':
        data_group = None
        for g in group:
            if g in data_grouped_as_index.groups.keys():
                tmp = pd.DataFrame({
                    "Group": [g for _ in range(len(data_grouped_as_index.get_group(g)))],
                    "Group_Name": ["_".join(g) for _ in range(len(data_grouped_as_index.get_group(g)))],
                    "Hive_number":  data_grouped_as_index.get_group(g)["Hive_number"],
                    variable: data_grouped_as_index.get_group(g)[variable]})
                if data_group is None:
                    data_group = tmp
                else:
                    data_group = pd.concat([data_group, tmp])     

    return data_group

def stat_decision_tree(data: pd.DataFrame, variable: str, group_columns: List[str]):
    """
    Decide what statistical test to perfom depending on the group size and returns associated p-values.
    
    Decision steps:
    ---------------
        1. Is there a differences in between all groups: returns one value for entire test
            a. At least one group has less than 0 and 20 elements: Kruskal-Wallis Test. Warn that test will be uncertain if less than 5 elements, it is better to remove this group.
            b. All groups have more than 20 elements: ANOVA test
        2. If there is a difference in between the groups, which ones are differents: returns one value for each combination of two groups
            a. At least one group with less than 20 elements: paired Mann-Whitney U test
            b. All groups have more than 20 elements: Shapiro test
                i. if all shapiro tests give p-value less than 5%:
                ii. if shapiro test has a least one-value above 5%: Kruskal-Wallis Test, if below 5%, then paired Mann-Withney U.
        3. If there are statistcal differences between groups, how big is this difference ? Size effect test
        
        1. Determine:
            a. Type if data: {continous, ordinal, nominal}
            b. If continious, determine if the data is normally distributed AND do not have extreme values
                a. Normal and no extreme values: parametric
                b. if the type of data AND if the groups are normally distributed AND do not have extrem values
            a. Continuous, normally distri
        
        2. 
            a. If yes, parametric test
            b. If no, non-parametric test
                
    """
    
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

def conservative_stat_decision_tree(data: pd.DataFrame, variable: str, group_columns: List[str], variable_type: str, what: str, day_dependance: str=None, force_parametric=False):
    """
    Decide what test to perform, generate report and returns associated p-values. All groups are considered as independant expect for 'Day' if day_dependance is set to 'match'.

    Parameters:
    -----------
        data: pandas DataFrame.
        Dataframe containing as header at least the variable to look at and the group_columns used to group the variable.

        variable: string.
        Name of the variable to look at. The exact same string should be a header from data.

        group_column: List of string.
        On what you want to group the variable. Each element of group_column should be a header from data.

        variable_type: str {'continuous', 'ordinal', 'nominal'}.
        The type of data you want to look at:
            -'continuous': variable that can have whatever value. It is usually the case for all data measured using an instrument.
            -'ordinal': categorical values that can be sorted (ex: )
            -'nominal': categorical values that cannot be sorted (ex: colors)

        what: str {'difference', 'relationship'}
        What you want to look at on your data:
            -'difference': want to determine if there is a significant difference in one variable of your samples
            -'relationship': want to determine if there is a relationship between 2 variables of your samples

        day_dependance: str {'match', 'anonymous'}
        If 'Day' is one of your grouping variable (ie, one element of group_column list), defines if 'Day' should be considered as dependant or independant variable.
            -'match': all of your samples have match paires --> 'Day' is considered as a dependant variable
            -'anonymous': your samples is not matching paires --> 'Day' is considered as an independant variable

        force_parametric: bool. False by default
        Set to True if you want to perform a parametric test eventhough the conditions to satisfy parametric test are not reached.


    Return:
    -------
        results: dict
        Result of the statistical tests for each 2group combination.
        results = {
            "what":
            "variable":
            "conditions":
            "variable_type":
            "day_dependance":
            "force_parametric":
            "stats":
            "Parametric":
            "Dependant":
            {
                "Test on groups: XX":
                {
                    "Global": if more than 2 groups
                    [{
                        "Ngroup":
                        "Global Test":
                        "P-value":
                    }]
                    "Comparisons":
                    [
                        {
                            "Test":
                            "Group":
                            "P-value":
                        },

                    ]
                }
            }
            }
        }

    Representation of decisison tree:
    ---------------------------------
    START
    │
    ├─ Objective?
    │   ├─ Difference → continue
    │   └─ Relationship → Pearson if param-eligible, else Spearman
    │
    ├─ Data type?
    │   ├─ Nominal → Chi² / Fisher / McNemar / Cochran Q → END
    │   ├─ Ordinal → Wilcoxon / Mann–Whitney / Friedman / Kruskal → END
    │   └─ Continuous → continue
    │
    ├─ Determine:
    │   ├─ Paired vs Independent
    │   ├─ 2 groups vs >2 groups
    │   ├─ n per group
    │   ├─ Skewness per group
    │   ├─ % outliers (1.5*IQR)
    │   └─ Extreme z > |3.5|
    │
    ├─ Parametric eligibility:
    │   IF (ALL groups satisfy):
    │       n ≥ 25
    │       AND |skew| ≤ 2
    │       AND <5% outliers
    │       AND no extreme z
    │   → PARAMETRIC
    │      ├─ 2 groups:
    │      │     Paired → Paired t-test
    │      │     Indep → t-test (Welch if unequal variance)
    │      │
    │      └─ >2 groups:
    │            Paired → RM-ANOVA → Paired t-test for each combination with Holm p-values adjustment
    │            Indep → ANOVA (Welch if unequal variance) → Tuckey for each combination (or Games-Howek if unequal variance)
    │
    └─ ELSE → NONPARAMETRIC
            ├─ 2 groups:
            │     Paired → Wilcoxon (Sign if very small n)
            │     Indep → Mann-Whitney
            └─ >2 groups:
                Paired → Friedman → Dunn Test with Holm Corrections
                Indep → Kruskal-Wallis → 
    """

    # What effect size to do and when:
    # 2 groups parametric → Hedges’ g
    # more than 2 groups → Omega squared (ω²)
    # Nonparametric → Cliff’s Delta
    # Correlation → report r or rho
    
    print("Conservative Decision Tree for Statistical Test")
    print("-----------------------------------------------")
    if "Day" in group_columns:
        print("3 set of tests will be performed:")
        print("\t1. Independant test for each day among the different samples")
        print("\t2. Dependant test for each sample among the different days")
        print("\t3. Depedant test for the dynamic difference among the different samples") #for this one do LMM (linear mixed models) for continuous data
    else:
        print("1 set of test will be performed: independant test among the different samples.")
    
    #Verify required headers are present
    for elt in (*group_columns, variable):
        check_header(elt, data.columns.values.tolist())
    if None in group_columns:
        group_columns.remove(None)

    stats = {"what": what, "variable": variable, "conditions": group_columns, 
             "variable_type": variable_type, "day_dependance": day_dependance, 
             "force_parametric": force_parametric}

    #Setting names of paired & unpaired columns from data
    paired_column = []
    if "Day" in group_columns:
        if day_dependance is None:
            raise KeyError(f"If 'Day' is used to group variable, 'day_dependance' should be set: either 'match' or 'anonymous'")
        elif day_dependance == 'match':
            paired_column.append("Day")
            paired_column_index = group_columns.index("Day")
            if paired_column_index != 0:
                raise KeyError(f"If 'Day' is used to group variable, 'Day' should be the first element of group_columns")
            unpaired_column = group_columns.copy()
            unpaired_column.remove("Day")
        elif day_dependance == 'anonymous':
            unpaired_column = group_columns.copy()
        else:
            raise KeyError(f"{day_dependance} key unkown for 'day_dependance'. Available keys are: 'match' and 'anonymous'.")

    #Setting the group names of paired & unpaired only
    unpaired_names = [list(set(data[elt])) for elt in unpaired_column]
    paired_names = [list(set(data[elt])) for elt in paired_column]
    unpaired_groups = list(itertools.product(*unpaired_names))
    paired_groups = list(itertools.product(*paired_names))

    #Setting and structuring the group names of paired and unpaired groups
    real_groups = list(data.groupby(by=group_columns).groups.keys())
    tmp_unpaired = [list(itertools.product([pairs], unpaired_groups)) for pairs in paired_groups]
    unpaired_groups_all = []
    for group_name in tmp_unpaired:
        unpaired_groups_all.append([sum(elt, ()) for elt in group_name if sum(elt, ()) in real_groups])
    tmp_paired = [list(itertools.product(paired_groups,[pairs])) for pairs in unpaired_groups]
    paired_groups_all = []
    for group_name in tmp_paired:
        paired_groups_all.append([sum(elt, ()) for elt in group_name if sum(elt, ()) in real_groups])
    print("Unpaired groups:", unpaired_groups_all)
    print("Paired groups:", paired_groups_all)

    if what == 'difference':

        if variable_type == 'ordinal':
            pass

        elif variable_type == "nominal":
            pass

        elif variable_type == 'continuous':
            
            #Compute basic stats per group
            data_grouped = data.groupby(by=group_columns, as_index=False)[variable]
            stat_group = data_grouped.describe()
            stat_group["IQR"] = stat_group["75%"] - stat_group["25%"] #adding IQR to stats
            stat_group["skew"] = data_grouped.skew()[variable] #adding skewness to stats
            stat_group["pct outlier"] = 0
            stat_group["Any(|Zscore|>3)"] = False
            stat_group["Any(value>3*std)"] = False
            i=0
            for name, group in data_grouped:
                print(group)
                below_5pct = (group < (stat_group["25%"][i] - 1.5*stat_group["IQR"][i])).value_counts().get(True)
                print("<5%", below_5pct)
                if below_5pct is None:
                    below_5pct = 0
                above_95pct = (group > (stat_group["75%"][i] + 1.5*stat_group["IQR"][i])).value_counts().get(True)
                if above_95pct is None:
                    above_95pct = 0
                stat_group.loc[i, "pct outlier"] = 100*(below_5pct + above_95pct) / stat_group["count"][i]
                stat_group.loc[i, "Any(|Zscore|>3)"] = any(abs(z) > 3 for z in zscore(group))
                print(name, group, 3*stat_group.loc[i, "std"],  (group > 3*stat_group.loc[i, "std"]).value_counts().get(True))
                # quit()
                stat_group.loc[i, "Any(value>3*std)"] = any((group > 3*stat_group.loc[i, "std"]))
                i+=1
            #add the test  on extrema
            stats["stats"] = stat_group.to_dict() #store statistical info
            stat_group["group"] = stat_group[group_columns].apply(tuple, axis=1)

            data_grouped_as_index = data.groupby(by=group_columns, as_index=True)

            # #Check if distributions are parametric
            # if all(stat_group["count"] >= 25) and all(abs(stat_group["skew"]) <= 2):
            #     stats["Parametric"] = True
            # else:
            #     stats["Parametric"] = False
                
            #Paired Test
            if len(paired_column) > 0:

                stats["Dependant"] = {}
                
                for group in paired_groups_all:

                    #Go to next iteration is one group do not exist
                    if any(elt not in data_grouped_as_index.groups.keys() for elt in group) or len(group) == 0:
                        continue

                    #Determine if the groups statifies conditions for parametric tests
                    sub_stat_group = stat_group.loc[stat_group["group"].isin(group)]
                    if all(abs(sub_stat_group["skew"]) < 2) and all(sub_stat_group["Any(|Zscore|>3)"] < 3) and all(sub_stat_group["pct outlier"] < 5):# and all(sub_stat_group["count"] >= 25):
                        parametric = True
                    else:
                        parametric = False

                    #Parametric Test
                    if  parametric == True or force_parametric == True: #add the condition on outliers and extrema

                        stats["Dependant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}                       
                        
                        #Only two groups ===> PAIRED T-TEST
                        if len(group) <= 2:    
                            data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')
                            _, pval = ttest_rel(*data_group)    
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 
                            stats["Dependant"][str(group)]["Comparisons"] = [{"Test": "Paired T-test", "Comparison": group, "P-value": pval,
                                                                              "N per group": n, "Warning": warn_mes}]     

                        #More than two groups ===> REPEATED MEASURES ANOVA
                        else:
                            data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'dataframe')
                            aov = AnovaRM(data_group, variable, "Hive_number", within=["Group_Name"]).fit()
                            pval_global = aov.anova_table['Pr > F']["Group_Name"]
                            stats["Dependant"][str(group)]["Global"] = [{"Ngroup": len(group), "Global Test": "Repeated Measures ANOVA", "P-value": pval_global}]
                            if pval_global < 0.05:
                                stats["Dependant"][str(group)]["Comparisons"] = []
                                group_combination = list(itertools.combinations(group, 2))
                                grouped = data_group.groupby(by="Group")
                                pval_no_adjust = []
                                for g in group_combination:
                                    gg = structure_data(grouped, g, variable, 'array_like')
                                    _, p = ttest_rel(*gg)
                                    pval_no_adjust.append(p)
                                pval = multipletests(pval_no_adjust, method='holm')
                                for i, g in enumerate(group_combination):
                                    n = (sub_stat_group.loc[sub_stat_group["group"] == g[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == g[1]]["count"].values[0])
                                    if any(elt < 15 for elt in n):
                                        warn_mes = "Less than 15 elements"
                                    else:
                                        warn_mes = None 
                                    stats["Dependant"][str(group)]["Comparisons"].append({
                                        "Test": "Paired T-test with Holm correction",
                                        "Group": g,
                                        "P-value": pval_no_adjust[i],
                                        "Adjusted P-values": pval[1][i],
                                        "N per group": n,
                                        "Warning": warn_mes
                                    })
                    
                    #Non parametric Test
                    else:

                        stats["Dependant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}
                        data_group = structure_data(data_grouped_as_index, group, variable, 'array_like')
                        
                        #Only two groups
                        if len(group) <= 2:    
                            
                            #The two groups have at least 10 elements each ===> WICOXON SIGNED RANK
                            if all(stat_group["count"] >= 10):
                                _, pval, _ = wilcoxon(*data_group)
                                test = "Wilconxon Signed Rank Test"
                        
                            #At least one group has less than 10 elements ===> SIGN TEST
                            else:
                                pval=None; test=None

                            #Verify number of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 

                            #Update results
                            stats["Dependant"][str(group)]["Comparisons"] = [{"Test": test, "Group": group, "P-value": pval,
                            "N per group": n, "Warning": warn_mes}]

                        #More than two groups ===> FRIEDMAN for global, then DUNN TEST and HOLM CORRECTIONS
                        else:
                            #Initiaing pair-wise outputs
                            gg = None
                            pval = None
                            #Perform Global Test: Friedman
                            _, pval_global = friedmanchisquare(*data_group)
                            stats["Dependant"][str(group)]["Global"] = [{"Ngroup": len(group), "Global Test": "Friedman", "P-value": pval_global}]
                            #Perform pair-wise tests: Dunn with Holm correcttion
                            if pval_global < 0.05:
                                dunn = posthoc_dunn(data_group, p_adjust='holm')
                                gg = []; pval=[]
                                stats["Dependant"][str(group)]["Comparisons"] = []
                                for i in dunn:
                                    for j in dunn.index:
                                        if j > i:
                                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[i-1]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j-1]]["count"].values[0])
                                            if any(elt < 15 for elt in n):
                                                warn_mes = "Less than 15 elements"
                                            else:
                                                warn_mes = None 
                                            stats["Dependant"][str(group)]["Comparisons"].append({"Test": "Dunn Test with holm corrections", 
                                                                                                  "Group":(group[i-1], group[j-1]), "P-value": dunn[i][j],
                                                                                                  "N per group": n, "Warning": warn_mes})

            #Unpaired Test
            if len(unpaired_column) > 0 and len(unpaired_groups) > 1:

                stats["Independant"] = {}

                for group in unpaired_groups_all:

                    if any(elt not in data_grouped_as_index.groups.keys() for elt in group) or len(group) == 0:
                        continue

                    print("Independant", group)

                    #Determine if the groups statifies conditions for parametric tests
                    sub_stat_group = stat_group.loc[stat_group["group"].isin(group)]
                    if all(abs(sub_stat_group["skew"]) < 2) and all(sub_stat_group["Any(|Zscore|>3)"]) < 3 and all(sub_stat_group["pct outlier"] < 5): #and all(sub_stat_group["count"] >= 25)
                        parametric = True
                    else:
                        parametric = False

                    stats["Independant"][str(group)] = {"Parametric": parametric, "Global": None, "Comparisons": None}

                    #Parametric Test
                    if parametric == True or force_parametric == True: 

                        data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')
                        _, variance = bartlett(*data_group)

                        #Only two groups
                        if len(group) <= 2:

                            #Equal variance ===> UNPAIRED T-TEST
                            if variance >= 0.05:
                                _, pval = ttest_ind(*data_group, equal_var=True)  
                                test = "Unpaired T-Test"    

                            #Unequal variance ===> WELCH T-TEST
                            else:
                                _, pval = ttest_ind(*data_group, equal_var=False)  
                                test = "Welch T-Test"  

                            #Determine number of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 

                            #Update result
                            stats["Independant"][str(group)]["Comparisons"] = [{
                                "Variance Test": "Barlett", 
                                "Variance P-Value": variance, 
                                "Group": group, "P-value": pval, "Test": test,
                                "N per group": n, "Warning": warn_mes}]   

                        #More than two groups: test variance
                        else:

                            #Equal variance ===> ONE-WAY ANOVA
                            if variance >= 0.05:
                                glob= "One-way ANOVA"
                                _, pval_global = f_oneway(*data_group, equal_var=True)
                                if pval_global < 0.05:
                                    tuk = tukey_hsd(*data_group, equal_var=True)
                                    test = "Pair-wise Tukey"
                                    stats["Independant"][str(group)]["Comparisons"] = []
                                    for i, elt in enumerate(tuk.pvalue):
                                        for j, p in enumerate(elt):
                                            if j > i:
                                                #Determine numebr of elements per group
                                                n = (sub_stat_group.loc[sub_stat_group["group"] == group[i]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j]]["count"].values[0])
                                                if any(elt < 15 for elt in n):
                                                    warn_mes = "Less than 15 elements"
                                                else:
                                                    warn_mes = None 
                                                #Update results
                                                stats["Independant"][str(group)]["Comparisons"].append({"Test": test, "Group": (group[i], group[j]), "P-value": p,
                                                                                                        "N per group": n, "Warning": warn_mes})    

                            #Unequal variance ===> WELCH ANOVA
                            else:
                                glob= "Welch ANOVA"
                                _, pval_global = f_oneway(*data_group, equal_var=False)
                                if pval_global < 0.05:
                                    tuk = tukey_hsd(*data_group, equal_var=False) 
                                    test = "Games-Howell"
                                    stats["Independant"][str(group)]["Comparisons"] = []
                                    for i, elt in enumerate(tuk.pvalue):
                                        for j, p in enumerate(elt):
                                            if j > i:
                                                #Determine numebr of elements per group
                                                n = (sub_stat_group.loc[sub_stat_group["group"] == group[i]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[j]]["count"].values[0])
                                                if any(elt < 15 for elt in n):
                                                    warn_mes = "Less than 15 elements"
                                                else:
                                                    warn_mes = None 
                                                #Update results
                                                stats["Independant"][str(group)]["Comparisons"].append({"Test": test, "Group": (group[i], group[j]), "P-value": p,
                                                                                                        "N per group": n, "Warning": warn_mes})                            

                            stats["Independant"][str(group)]["Global"] = [{"Ngroup": len(group), "Variance Test": "Bartlett", "Variance P-value": variance,
                                                                "Global Test": glob, "P-value": pval_global}]

                    #Non-parametric Test
                    else:

                        data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'array_like')

                        #Only two groups ===> MANN-WHITNEY
                        if len(group) <= 2:
                            #Statistic Test  
                            _, pval = mannwhitneyu(*data_group)
                            #Determine numebr of elements per group
                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[0]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[1]]["count"].values[0])
                            if any(elt < 15 for elt in n):
                                warn_mes = "Less than 15 elements"
                            else:
                                warn_mes = None 
                            #Update results
                            stats["Independant"][str(group)]["Comparisons"] = [{
                                "Test": "Mann-Whitney U", "Group": group, "P-value": pval,
                                "N per group": n, "Warning": warn_mes}]   

                        #More than two groups ===> KRUSKAL-WALLIS for global, than DUNN TEST and HOLM CORRECTIONS
                        else:
                            _, pval_global = kruskal(*data_group)
                            stats["Independant"][str(group)]["Global"] = [{
                                "Ngroup": len(group), "Test": "Kruskal-Wallis", "P-value": pval_global}]
                            if pval_global <0.05:
                                data_group = structure_data(data_grouped_as_index[[variable, "Hive_number"]], group, variable, 'dataframe')
                                dunn = posthoc_dunn(data_group, val_col=variable, group_col="Group_Name", p_adjust='holm')
                                stats["Independant"][str(group)]["Comparisons"] = []
                                for ii, i in enumerate(dunn):
                                    for jj, j in enumerate(dunn.index):
                                        if j > i:
                                            #Determine numebr of elements per group
                                            n = (sub_stat_group.loc[sub_stat_group["group"] == group[ii-1]]["count"].values[0], sub_stat_group.loc[sub_stat_group["group"] == group[jj-1]]["count"].values[0])
                                            if any(elt < 15 for elt in n):
                                                warn_mes = "Less than 15 elements"
                                            else:
                                                warn_mes = None 
                                            #Update results
                                            stats["Independant"][str(group)]["Comparisons"].append({"Test": "Dunn Test with Holm corrections", 
                                                                                                  "Group":(group[ii-1], group[jj-1]), "P-value": dunn[i][j],
                                                                                                  "N per group": n, "Warning": warn_mes})   



    return stats

#Read user inputs
graph_user_inputs = read_usr_input("graph_inputs.txt")
print(graph_user_inputs)
if len(graph_user_inputs["Graph"]) == 0:
    print("No graph requested.")

#Open, Combine & Save results
data = open_results(graph_user_inputs, path_delimiter)
print("Data loaded:\n", data.columns.values)
# data.to_csv(graph_user_inputs["Outpath"])

#Add Day number
tmp = data["Day"].replace("D", "", regex=True)
data["Day Number"] = tmp
data["Day Number"] = data["Day Number"].astype(int) #will create issue if - are present !!!!!!!
del(tmp)

data["Hive_number"] = data["Hive number"]
data = pd.concat([data.loc[data["Hive_number"] == 0], data.loc[data["Hive_number"] == 3], data.loc[data["Hive_number"] == 4]])

# data_group = data.groupby(by=['Day', 'Substrate', 'Condition'], as_index=True)
# print(data_group.groups.keys())
# group = [('D6', 'H2M100-ShortFlip', 'Exp006-B'),('D11', 'H2M100-ShortFlip', 'Exp006-B'), ('D18', 'H2M100-ShortFlip', 'Exp006-B')]#, ('D18', 'H2M100-ShortFlip', 'Exp006-B')]
# grouped = structure_data(data_group, group, "area_filled", 'array_like')
# tuk = tukey_hsd(*grouped)
# print(tuk, tuk.pvalue)
# # group = [('D11', 'H2M100-ShortFlip', 'Exp006-B'), ('D18', 'H2M100-ShortFlip', 'Exp006-B')]
# # grouped = structure_data(data_group, group, "area_filled", 'array_like')
# # print(grouped)
# # grouped = [[grouped[0][0], grouped[0][2], grouped[0][3]], [grouped[1][0], grouped[1][4], grouped[1][5]]]
# # stat, pval = ttest_rel(*grouped)
# quit()


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

        # #Stats
        # stat_decision_tree(data, y[i_plot], [x[i_plot], z, *conditions])
                

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
#plots(data, ["Day"], ["eccentricity"], ["Condition"], ["boxplot"])

area_normal = [9.5, 9.75, 10, 10.25, 10.5]
area_poisson = [9.8, 9.85, 9.9, 10, 10.1, 10.15, 10.2, 15, 20, 21]
area_normal_heavy_tails = [1, 5, 9.5, 9.8, 9.9, 10, 10.2, 10.5, 15, 20]
area = area_normal_heavy_tails
#print(np.median(area), median_abs_deviation(area), np.mean(area), np.std(area), np.percentile(area, 25),  np.percentile(area, 75), zscore(area), skew(area))
#plots(data, ["Day"], ["area_filled"], ["Substrate"], ["boxplot"])
# area = area_poisson
# print(np.median(area), median_abs_deviation(area), np.mean(area), np.std(area), np.percentile(area, 25),  np.percentile(area, 75), zscore(area), skew(area))

data = pd.DataFrame({
    "Hive_number": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]*9,
    "Substrate": ["S1"]*10 + ["S2"]*10 + ["S3"]*10 + ["S1"]*10 + ["S2"]*10 + ["S3"]*10 +  ["S1"]*10 + ["S2"]*10 + ["S3"]*10,
    "Day": ["D1"]*30 + ["D4"]*30 + ["D8"]*30,
    "area_filled": area + [elt*2 for elt in area] + [elt*1.1 for elt in area] + [elt*3 for elt in area] + [elt*8 for elt in area]  + [elt*3.3 for elt in area] + [elt*4 for elt in area] + [elt*10 for elt in area]  + [elt*4.4 for elt in area],
    "Condition": ["c1"]*10 + ["c2"]*10 + ["c1"]*10 + ["c1"]*10 + ["c2"]*10  + ["c1"]*10 + ["c1"]*10 + ["c2"]*10  + ["c1"]*10,
    "Manually Discarded": [False]*90
})
# data = pd.DataFrame({
#     "Hive_number": [0, 1, 2, 3, 4]*6,
#     "Substrate": ["S1"]*5 + ["S2"]*5 + ["S1"]*5 + ["S2"]*5 +  ["S1"]*5 + ["S2"]*5,
#     "Day": ["D1"]*10 + ["D4"]*10 + ["D8"]*10,
#     "area_filled": area + [elt*2 for elt in area] + [elt*3 for elt in area] + [elt*8 for elt in area]  + [elt*4 for elt in area] + [elt*10 for elt in area],
#     "Condition": ["c1"]*5 + ["c2"]*5 + ["c1"]*5 + ["c2"]*5  + ["c1"]*5 + ["c2"]*5
# })
print(data)

stats = conservative_stat_decision_tree(data, "area_filled", ["Day", "Substrate", "Condition"], 'continuous', 'difference', day_dependance='match', force_parametric=True)
pretty_print(stats)
plots(data, ["Day"], ["area_filled"], ["Substrate"], ["boxplot"])

