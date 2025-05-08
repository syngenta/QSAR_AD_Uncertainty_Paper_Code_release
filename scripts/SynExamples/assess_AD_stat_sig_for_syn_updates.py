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
##############################
#Copright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import os,sys,time,pickle,re
from collections import defaultdict
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(dir_of_this_script)
#----------------------------------------------
sys.path.append(top_dir)
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from consistent_parameters_for_all_modelling_runs import consistent_no_random_splits_for_AD_p_val
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
#------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
#----------------------------------------------------------------------------
#-------------
reg_top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData','logP_updates'])
reg_top_calc_dir = os.path.sep.join([reg_top_ds_dir,'Calc'])
reg_all_raw_pkl = os.path.sep.join([reg_top_calc_dir,'logP.All.Updates.Raw.pkl'])
#------------
class_top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData','DT50_Updates'])
class_top_calc_dir = os.path.sep.join([class_top_ds_dir,'Calc'])
class_all_raw_pkl = os.path.sep.join([class_top_calc_dir,'DT50.All.Updates.Raw.pkl'])
#-----------
endpoint_type_matched_to_pkl_of_timepoints_with_csvs_of_raw_res_in_and_out_ad = {}
endpoint_type_matched_to_pkl_of_timepoints_with_csvs_of_raw_res_in_and_out_ad['Regression']=reg_all_raw_pkl
endpoint_type_matched_to_pkl_of_timepoints_with_csvs_of_raw_res_in_and_out_ad['Classification']=class_all_raw_pkl
#-------------

def loadResults(raw_res_pkl,ep_type_matched_to_default_AD_uncertainty_methods,type_of_modelling):
    f_in = open(raw_res_pkl,'rb')
    try:
        allRawResDicts = pickle.load(f_in)
    finally:
        f_in.close()
    
    AD_method_name = ep_type_matched_to_default_AD_uncertainty_methods[type_of_modelling]['AD_method_name']
    
    method = ep_type_matched_to_default_AD_uncertainty_methods[type_of_modelling]['uncertainty_method']
    
    dict_for_all_test_sets_of_raw_subset_results = defaultdict(dict)
    
    for test_set_name in allRawResDicts.keys():
        for rand_seed_label in allRawResDicts[test_set_name].keys():
            dict_for_all_test_sets_of_raw_subset_results[test_set_name][rand_seed_label] = allRawResDicts[test_set_name][rand_seed_label][AD_method_name][method]

            #We need to avoid this: Exception: There should only be two subset names, e.g. "inside AD" and "outside AD". But, the raw results dictionary you have provided contains names for these subsets=['All', 'Inside', 'Outside']
            del dict_for_all_test_sets_of_raw_subset_results[test_set_name][rand_seed_label]['All'] 
    
    return dict_for_all_test_sets_of_raw_subset_results

def main():
    print('THE START')
    
    ########################
    subset_1_name='Inside'
    subset_2_name='Outside'
    no_rand_splits=consistent_no_random_splits_for_AD_p_val
    ########################
    
    for type_of_modelling in endpoint_type_matched_to_pkl_of_timepoints_with_csvs_of_raw_res_in_and_out_ad.keys():
               
        raw_res_pkl = endpoint_type_matched_to_pkl_of_timepoints_with_csvs_of_raw_res_in_and_out_ad[type_of_modelling]
        
        dict_for_all_test_sets_of_raw_subset_results = loadResults(raw_res_pkl,ep_type_matched_to_default_AD_uncertainty_methods,type_of_modelling)
        
        #==================
        if 'Classification' == type_of_modelling:
            out_dir = os.path.sep.join([class_top_calc_dir,'AD.stat.sig'])
            

            _type_of_modelling = 'binary_class'
            metric_to_expected_sub_1_minus_sub_2_sign = classifcation_metric_to_expected_sub_1_minus_sub_2_sign
        elif 'Regression' == type_of_modelling:
            out_dir = os.path.sep.join([reg_top_calc_dir,'AD.stat.sig'])

            _type_of_modelling = 'regression'
            metric_to_expected_sub_1_minus_sub_2_sign = regression_metric_to_expected_sub_1_minus_sub_2_sign
        else:
            raise Exception(f'Unrecognised type_of_modelling={type_of_modelling}')
        #===================

        
        
        createOrReplaceDir(dir_=out_dir)

        workflowForAFamilyOfRawResults(out_dir,dict_for_all_test_sets_of_raw_subset_results,subset_1_name=subset_1_name,subset_2_name=subset_2_name,
                                   type_of_modelling=_type_of_modelling,no_rand_splits=no_rand_splits,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=list(metric_to_expected_sub_1_minus_sub_2_sign.keys()),x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=True,metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,p_vals_table=f'one_tail_{type_of_modelling}_PVals.csv',
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=True,
                                   debug=True, scenarios_with_errors=[],set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True)
        
        workflowForAFamilyOfRawResults(out_dir,dict_for_all_test_sets_of_raw_subset_results,subset_1_name=subset_1_name,subset_2_name=subset_2_name,
                                   type_of_modelling=_type_of_modelling,no_rand_splits=no_rand_splits,strat_rand_split_y_name='strat_y',rand_seed=42,
                                   metrics_of_interest=list(metric_to_expected_sub_1_minus_sub_2_sign.keys()),x_lab='Groups',y_lab='Shift-Metric',legend_lab='Split Basis',sig_level_perc=5,
                                   one_sided_sig_test=False,metric_to_expected_sub_1_minus_sub_2_sign=None,p_vals_table=f'{type_of_modelling}_PVals.csv',
                                   adjusted_p_vals_table_name=None,results_are_obtained_over_multiple_folds_and_or_seeds=True,create_plots=True,
                                   debug=False, scenarios_with_errors=[],set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True)
        
        
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
