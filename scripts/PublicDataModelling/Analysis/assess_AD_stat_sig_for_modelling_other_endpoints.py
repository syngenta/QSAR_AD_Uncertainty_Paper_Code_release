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
import os,sys,glob,re
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
##################################
from assess_AD_stat_sig_for_exemplar_public_data import add_dummy_values_for_missing_subset_results
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,createOrReplaceDir,doubleDefaultDictOfLists,load_from_pkl_file
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import raw2prettyMetricNames,regression_stats_in_desired_order,classification_stats_in_desired_order,stats_metadata_cols_in_desired_order
from consistent_parameters_for_all_modelling_runs import ADMethod2Param
from PublicDataModelling.general_purpose.common_globals import regression_dataset_names,ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list,get_ds_matched_to_other_endpoints_of_interest
from PublicDataModelling.general_purpose.common_globals import top_class_or_reg_ds_dirs
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
from consistent_parameters_for_all_modelling_runs import classification_stats_in_desired_order,regression_stats_in_desired_order
from consistent_parameters_for_all_modelling_runs import endpoint_col,test_set_type_col,fold_col,rnd_seed_col,alg_col,ad_col,ad_subset_col
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from consistent_parameters_for_all_modelling_runs import consistent_no_random_splits_for_AD_p_val
from PublicDataModelling.Analysis.check_missing_or_inf_metrics_are_for_the_expected_reasons import get_fold_from_test_set_label,get_test_name_ignoring_fold_endpoint_prefix_fps_suffix_rnd_label_from_test_set_label
#-------------
out_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','Other_EPs_AD_P_vals'])
modelling_results_subdir='Modelling.2'

def identify_all_relevant_pkl_files(type_of_modelling,top_class_or_reg_ds_dirs,modelling_results_subdir,ADMethod2Param=ADMethod2Param,regression_dataset_names=regression_dataset_names):
    all_pkl_files = []

    for ds in top_class_or_reg_ds_dirs.keys():
        #=======================
        if "Classification" == type_of_modelling:
            if ds in regression_dataset_names:
                continue
        elif "Regression" == type_of_modelling:
            if not ds in regression_dataset_names:
                continue
        else:
            raise Exception(f'Unexpected type of modelling={type_of_modelling}')
        #=========================

        
        other_endpoints_of_interest = get_ds_matched_to_other_endpoints_of_interest(ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list)[ds]
        
        for ep in other_endpoints_of_interest:
            all_pkl_files += [os.path.sep.join([top_class_or_reg_ds_dirs[ds],modelling_results_subdir,f'{ep}_k={ADMethod2Param["UNC"]}_t={ADMethod2Param["Tanimoto"]}','RawResAndStats.pkl'])]
        
    
    return all_pkl_files

def reorganize_raw_results_from_pkl_files(all_pkl_files,type_of_modelling,subset_1_name,subset_2_name):
   
    #########################
    ##see modelsADuncertaintyPkg\qsar_eval\assess_stat_sig_shift_metrics.py comments:
    #dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name] = dict_for_all_folds_and_or_model_seeds_of_raw_subset_results
    #dict_for_all_folds_and_or_model_seeds_of_raw_subset_results[fold_and_or_model_seed_combination] = dict_of_raw_subset_results
    #See def getDictOfMergedIdsMatched2RawResults(...): for a description of dict_of_raw_subset_results
    
    dict_for_all_test_sets_of_raw_subset_results = neverEndingDefaultDict()
    ###########################

    seen_endpoints = []

    for pkl_file in all_pkl_files:
        dict_of_raw_results = load_from_pkl_file(pkl_file)['dict_of_raw_results']

        
        for endpoint in dict_of_raw_results.keys():
			#===============================================
            if endpoint in seen_endpoints: raise Exception(f'Endpoint names should be unique across datasets! - But {endpoint} occurs twice!')

            seen_endpoints.append(endpoint)
			#===============================================

            for test_set_label in dict_of_raw_results[endpoint].keys():
                for rand_seed in dict_of_raw_results[endpoint][test_set_label].keys():
                    for AD_method_name in dict_of_raw_results[endpoint][test_set_label][rand_seed].keys():
                        for method in dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name].keys():
                            test_name_ignoring_fold = get_test_name_ignoring_fold_endpoint_prefix_fps_suffix_rnd_label_from_test_set_label(test_set_label,endpoint)

                            fold = get_fold_from_test_set_label(test_set_label,type_of_modelling)

                            test_set_plus_methods_name = f'{endpoint}_{test_name_ignoring_fold}_AD={AD_method_name}_m={method}'

                            fold_and_or_model_seed_combination = f'f={fold}_s={rand_seed}'

                            dict_of_raw_subset_results = defaultdict(dict)

                            for AD_subset in dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method].keys():
                                #-----------------------
                                if not AD_subset in [subset_1_name,subset_2_name]: continue
                                #-----------------------

                                top_level_dict = dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]

                                subset_test_ids = top_level_dict['subset_test_ids']
                                subset_test_y = top_level_dict['subset_test_y']
																							
                                dict_of_raw_subset_results[AD_subset]['subset_test_ids'] = subset_test_ids
                                dict_of_raw_subset_results[AD_subset]['subset_test_y'] = subset_test_y
                                
                                if 'Regression' == type_of_modelling:
                                    subset_test_predictions = top_level_dict['subset_test_predictions']
                                    subset_sig_level_2_prediction_intervals = top_level_dict['subset_sig_level_2_prediction_intervals']
                                    
                                    dict_of_raw_subset_results[AD_subset]['subset_test_predictions'] = subset_test_predictions
                                    dict_of_raw_subset_results[AD_subset]['subset_sig_level_2_prediction_intervals'] = subset_sig_level_2_prediction_intervals
                                elif 'Classification' == type_of_modelling:
                                    
                                   subset_probs_for_class_1 =  top_level_dict['subset_probs_for_class_1']
                                   subset_predicted_y = top_level_dict['subset_predicted_y']

                                   dict_of_raw_subset_results[AD_subset]['subset_probs_for_class_1'] = subset_probs_for_class_1
                                   dict_of_raw_subset_results[AD_subset]['subset_predicted_y'] = subset_predicted_y
                                else:
                                    raise Exception(f'Unrecognised type_of_modelling={type_of_modelling}')

                                                        
							
                                dict_for_all_test_sets_of_raw_subset_results[test_set_plus_methods_name][fold_and_or_model_seed_combination] = dict_of_raw_subset_results

    return dict_for_all_test_sets_of_raw_subset_results

def get_dict_for_all_test_sets_of_raw_subset_results(type_of_modelling,top_class_or_reg_ds_dirs,modelling_results_subdir,subset_1_name,subset_2_name,regression_dataset_names=regression_dataset_names):

    all_pkl_files = identify_all_relevant_pkl_files(type_of_modelling,top_class_or_reg_ds_dirs,modelling_results_subdir)

    dict_for_all_test_sets_of_raw_subset_results = reorganize_raw_results_from_pkl_files(all_pkl_files,type_of_modelling,subset_1_name,subset_2_name)

    for ds in top_class_or_reg_ds_dirs.keys():
        #=======================
        if "Classification" == type_of_modelling:
            if ds in regression_dataset_names:
                continue
        elif "Regression" == type_of_modelling:
            if not ds in regression_dataset_names:
                continue
        else:
            raise Exception(f'Unexpected type of modelling={type_of_modelling}')
        #=========================

        dict_for_all_test_sets_of_raw_subset_results = add_dummy_values_for_missing_subset_results(context_mapped_to_dict_for_p_value_calculation=dict_for_all_test_sets_of_raw_subset_results,dataset_name=ds)

    return dict_for_all_test_sets_of_raw_subset_results


def main():

    ########################
    subset_1_name='Inside'
    subset_2_name='Outside'
    no_rand_splits=consistent_no_random_splits_for_AD_p_val
    ########################

    print('THE START')

    createOrReplaceDir(out_dir)

    for type_of_modelling in ep_type_matched_to_default_AD_uncertainty_methods.keys():
        print(f'Computing raw and locally adjusted p-values for other endpoints from the {type_of_modelling} datasets ...')

        dict_for_all_test_sets_of_raw_subset_results = get_dict_for_all_test_sets_of_raw_subset_results(type_of_modelling,top_class_or_reg_ds_dirs,modelling_results_subdir,subset_1_name,subset_2_name)

        #==================
        if 'Classification' == type_of_modelling:
            _type_of_modelling = 'binary_class'
            metric_to_expected_sub_1_minus_sub_2_sign = classifcation_metric_to_expected_sub_1_minus_sub_2_sign
        elif 'Regression' == type_of_modelling:
            _type_of_modelling = 'regression'
            metric_to_expected_sub_1_minus_sub_2_sign = regression_metric_to_expected_sub_1_minus_sub_2_sign
        else:
            raise Exception(f'Unrecognised type_of_modelling={type_of_modelling}')
        #===================


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

