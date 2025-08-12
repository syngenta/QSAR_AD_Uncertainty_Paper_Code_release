#Syngenta Open Source release: This file is part of code developed in the context of a Syngenta funded collaboration with the University of Sheffield: "Improved Estimation of Prediction Uncertainty Leading to Better Decisions in Crop Protection Research". In some cases, this code is a derivative work of other Open Source code. Please see under "If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below" for further details.
#Copyright (c) 2021-2025  Syngenta
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#Contact: richard.marchese_robinson [at] syngenta.com
#==========================================================
#If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below
#==========================================================
############################################################
#Copyright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys,re
import pandas as pd
from collections import defaultdict


this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
from consistent_parameters_for_all_modelling_runs import classifcation_metric_to_expected_sub_1_minus_sub_2_sign,regression_metric_to_expected_sub_1_minus_sub_2_sign
all_relevant_classification_statistics_names = [s for s in classifcation_metric_to_expected_sub_1_minus_sub_2_sign.keys()]
all_relevant_regression_statistics_names = [s for s in regression_metric_to_expected_sub_1_minus_sub_2_sign.keys()]
sys.path.append(os.path.sep.join([this_dir,'PublicDataModelling','Analysis']))
from AD_ranking import compute_shift_metrics

top_res_dir = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))



all_relevant_class_stats_csvs = []
all_relevant_class_stats_csvs.append(os.path.sep.join([top_res_dir,'PublicData','AllMergedStats','All_Endpoints_Classification_Stats.csv']))
all_relevant_class_stats_csvs.append(os.path.sep.join([top_res_dir,'SyngentaData','Merged_Stats','SYN_DT50_Classification_Stats.csv']))

all_relevant_reg_stats_csvs = []
all_relevant_reg_stats_csvs.append(os.path.sep.join([top_res_dir,'PublicData','AllMergedStats','All_Endpoints_Regression_Stats.csv']))
all_relevant_reg_stats_csvs.append(os.path.sep.join([top_res_dir,'SyngentaData','Merged_Stats','SYN_logP_Regression_Stats.csv']))

def get_overall_limit(dfs_list,stat_name,limit_func):

    all_vals = []

    for df in dfs_list:
        all_vals += [v for v in df[stat_name].tolist() if not pd.isna(v)]
    
    return limit_func(all_vals)



def get_overall_min_or_max(csv_list,stat_name,limit='min',shift_or_raw='raw'):
    
    if 'min' == limit:
        limit_func = min
    elif 'max' == limit:
        limit_func = max
    else:
        raise Exception(f'Unrecognised limit={limit}')
    
    dfs_list = [pd.read_csv(csv_file) for csv_file in csv_list]

    if 'raw' == shift_or_raw:
        metric_col_name = stat_name
    elif 'shift' == shift_or_raw:
        dfs_list = [compute_shift_metrics(df,metrics=[stat_name]) for df in dfs_list]
        #Hardcoded in def compute_shift_metrics(...):
        metric_col_name = f"{stat_name} Difference"
    else:
        raise Exception(f'Urencognised shift_or_raw option={shift_or_raw}')
    
    
    return get_overall_limit(dfs_list,metric_col_name,limit_func)



globally_observed_min_max_metric_and_shifts_dict = defaultdict(dict)

for modelling_type in ['Classification','Regression']:
    if 'Classification' == modelling_type:
        stats_names_list = all_relevant_classification_statistics_names
        csv_list = all_relevant_class_stats_csvs
    elif 'Regression' == modelling_type:
        stats_names_list = all_relevant_regression_statistics_names
        csv_list = all_relevant_reg_stats_csvs
    else:
        raise Exception(f'Unexpected modelling_type={modelling_type}')
    
    for limit in ['min','max']:
        for shift_or_raw in ['raw','shift']:
            for stat_name in stats_names_list:
                if 'shift' == shift_or_raw:
                    name_to_check_in_dict = f'Shift:{stat_name}'
                elif 'raw' == shift_or_raw:
                    name_to_check_in_dict = stat_name
                else:
                    raise Exception(f'Unexpected shift_or_raw={shift_or_raw}')

                globally_observed_min_max_metric_and_shifts_dict[name_to_check_in_dict][limit] = get_overall_min_or_max(csv_list,stat_name,limit,shift_or_raw)



