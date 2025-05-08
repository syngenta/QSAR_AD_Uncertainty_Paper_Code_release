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
import numpy as np
from collections import defaultdict
#-----------------
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
#----------------------------------------------------------------------------
top_level_of_public_data_scripts_dir = os.path.dirname(dir_of_this_script)
sys.path.append(os.path.sep.join([top_level_of_public_data_scripts_dir,'general_purpose']))
from common_globals import top_class_or_reg_ds_dirs,regression_dataset_names
#----------------------------------------------------------------------------
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict,reportSubDirs,createOrReplaceDir,load_from_pkl_file,convertDefaultDictDictIntoDataFrame
from modelsADuncertaintyPkg.qsar_eval.enforce_minimum_no_instances import size_of_inputs_for_stats_is_big_enough
from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest
from modelsADuncertaintyPkg.qsar_eval.reg_perf_pred_stats import coeffOfDetermination,rmse
from modelsADuncertaintyPkg.qsar_eval.all_key_class_stats_and_plots import get_experi_class_1_probs
#----------------------------------------------------------------------------
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import raw2prettyMetricNames,regression_stats_in_desired_order,classification_stats_in_desired_order,stats_metadata_cols_in_desired_order
from consistent_parameters_for_all_modelling_runs import lit_precedent_delta_for_calib_plot,sig_level_of_interest
#-------------
out_dir_of_merge_stats_files = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','ModellingResSummary'])
dict_of_merged_stats_files = {}
for dataset_type in ['Classification','Regression']:
    dict_of_merged_stats_files[dataset_type] = os.path.sep.join([out_dir_of_merge_stats_files,f'Exemplar_Endpoints_{dataset_type}_Stats.csv'])

#######################
#see calls to size_of_inputs_for_stats_is_big_enough(....) in all_key_reg_stats_and_plots.py and all_key_class_stats_and_plots.py
consistent_min_no_cmpds_for_stats = 2
#######################

metric_to_criteria_to_be_missing = defaultdict(list)
#----
too_few = 'Too few compounds'
zero_one_class = 'Zero compounds in one class'
zero_pos_class = 'Zero compounds in +ve class'
zero_neg_class = 'Zero compounds in -ve class'
zero_pos_pred = 'Zero compounds predicted to belong to +ve class'
zero_neg_pred = 'Zero compounds predicted to belong to -ve class'
const_preds = 'Constant predictions'
prob_range_lt_delta = 'Range of estimated class-1 probabilities is less than delta'
zero_width_PIs = 'Zero-width Prediction Intervals'
const_width_PIs = 'Constant-width Prediction Intervals'
############
#Constant experimental values for classification dataset are already handled by 'zero_one_class'
const_reg_experi = 'Experimental values for regression dataset are constant' 
############
#----
standard_criteria = [too_few]
overall_class_criteria = [zero_one_class]
pos_class_criteria = [zero_pos_class]
neg_class_criteria = [zero_neg_class]
pred_pos_criteria = [zero_pos_pred]
pred_neg_criteria = [zero_neg_pred]
const_preds_criteria = [const_preds]
const_reg_experi_criteria = [const_reg_experi] 
spearman_pi_vs_residuals_criteria = [zero_width_PIs,const_width_PIs]
############################
#If the difference between the maximum and minimum estimated probabilities for 'class 1' is less than the delta value used for the delta-calibration plot, all test set compounds will lie inside the bin (+/-delta) used to estimate the observed probabilities for each compound, meaning these observed probabilties will be constant, which will result in various metrics being undefined!
certain_class_uncertainty_metrics_criteria = [prob_range_lt_delta] 
#############################
for metric in regression_stats_in_desired_order+classification_stats_in_desired_order:
    metric_to_criteria_to_be_missing[metric] += standard_criteria

    if metric in ['Balanced Accuracy','MCC','AUC','Kappa','Stratified Brier Score','R2 (cal)','Pearson coefficient (cal)','Spearman coefficient (cal)']:
        metric_to_criteria_to_be_missing[metric] += overall_class_criteria
    if metric in ['Recall (class 1)']:
        metric_to_criteria_to_be_missing[metric] += pos_class_criteria
    if metric in ['Recall (class 0)']:
        metric_to_criteria_to_be_missing[metric] += neg_class_criteria
    if metric in ['Precision (class 1)']:
        metric_to_criteria_to_be_missing[metric] += pred_pos_criteria
    if metric in ['Precision (class 0)']:
        metric_to_criteria_to_be_missing[metric] += pred_neg_criteria
    if metric in ['Pearson coefficient','Spearman coefficient']:
        metric_to_criteria_to_be_missing[metric] += const_preds_criteria
    if metric in ['R2 (cal)',	'Pearson coefficient (cal)','Spearman coefficient (cal)']:
        metric_to_criteria_to_be_missing[metric] += certain_class_uncertainty_metrics_criteria
    if metric in ['Spearman coefficient (PIs vs. residuals)']:
        metric_to_criteria_to_be_missing[metric] += spearman_pi_vs_residuals_criteria
    if metric in ['R2','Pearson coefficient','Spearman coefficient']:
        metric_to_criteria_to_be_missing[metric] += const_reg_experi_criteria
    

    
metric_to_criteria_to_be_inf = {'ENCE':[zero_width_PIs]}

def check_missing_metrics(scenario_to_missing_metrics_dict,scenario_to_raw_res_dict,metric_to_criteria_to_be_missing,scenario_to_cmpd_counts_dict):
    for scenario in scenario_to_missing_metrics_dict.keys():

        if scenario in scenario_to_raw_res_dict.keys():

            #############################
            #values which are not relevant for either classification or regression scenarios will be set to None in scenarios_to_raw_res_dict:

            if not scenario_to_raw_res_dict[scenario]['class_1_probs'] is None:

                class_1_probs = [float(v) for v in scenario_to_raw_res_dict[scenario]['class_1_probs']]
            else:
                class_1_probs = None

            reg_pred_intervals = scenario_to_raw_res_dict[scenario]['reg_pred_intervals']
            #############################

            predictions = scenario_to_raw_res_dict[scenario]['predictions']
            experi_vals = scenario_to_raw_res_dict[scenario]['experi_vals']
        else:
            assert 0 == scenario_to_cmpd_counts_dict[scenario],scenario

            class_1_probs = []
            reg_pred_intervals = []
            predictions = []
            experi_vals = []



        for missing_metric in scenario_to_missing_metrics_dict[scenario]:
            expected_to_be_missing = False

            reasons_why_this_could_be_missing = metric_to_criteria_to_be_missing[missing_metric]

            for reason in reasons_why_this_could_be_missing:
                if too_few == reason:
                    if not size_of_inputs_for_stats_is_big_enough(experi_vals,predictions,limit=consistent_min_no_cmpds_for_stats):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif const_reg_experi == reason:
                    if 1 == len(set(experi_vals)):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_one_class == reason:
                    if 0 == len([v for v in experi_vals if 1==v]) or 0 == len([v for v in experi_vals if 0==v]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_pos_class == reason:
                    if 0 == len([v for v in experi_vals if 1==v]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_neg_class == reason:
                    if 0 == len([v for v in experi_vals if 0==v]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_pos_pred == reason:
                    if 0 == len([v for v in predictions if 1==v]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_neg_pred == reason:
                    if 0 == len([v for v in predictions if 0==v]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif const_preds == reason:
                    if 1 == len(set(predictions)):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif prob_range_lt_delta == reason:
                    experi_probs = get_experi_class_1_probs(class_1_probs,experi_vals,lit_precedent_delta_for_calib_plot)
                    print(f'experi_probs={experi_probs}')
                    if 1 == len(set(experi_probs)):
                        if not (max(class_1_probs)-min(class_1_probs))<lit_precedent_delta_for_calib_plot:
                            print(f'More complicated reason for {missing_metric} to be missing based upon constant experimental class-1 probability estimates, for scenaro = {scenario}, class_1_probs={class_1_probs}, experi_vals={experi_vals}')
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif zero_width_PIs == reason:
                    if all([0==v for v in getIntervalsWidthOfInterest(sig_level_2_prediction_intervals={sig_level_of_interest:reg_pred_intervals},sig_level_of_interest=sig_level_of_interest).tolist()]):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                elif const_width_PIs == reason:
                    if 1 == len(set(getIntervalsWidthOfInterest(sig_level_2_prediction_intervals={sig_level_of_interest:reg_pred_intervals},sig_level_of_interest=sig_level_of_interest).tolist())):
                        expected_to_be_missing = True
                        print(f'Reason for missing metric = {reason} [metric = {missing_metric}, scenario = {scenario}')
                        break
                else:
                    raise Exception(f'Unrecognised reason for metric value to be missing: {reason}')
                
            ##################
            #Debugging:
            #R2 (cal) was genuinely missing from original (pre-merged) statistics CSV for the following scenario, even though RMSE (cal) was present and consistent with merged statistics file
            #try:
            assert expected_to_be_missing,f'Unexpected missing metric {missing_metric}, for scenario = {scenario}, class_1_probs={class_1_probs}, predictions={predictions}, experi_vals={experi_vals}, reg_pred_intervals={reg_pred_intervals}'
            #except AssertionError as err:
            #    print(err)
            #    print(f'Recomputed RMSE (cal) = {rmse(e=experi_probs,p=class_1_probs)}')
            #    raise Exception(f'Missing R2 SHOULD BE = {coeffOfDetermination(e=experi_probs,p=class_1_probs)}')




def check_inf_metrics(scenario_to_inf_metrics_dict,scenario_to_raw_res_dict,metric_to_criteria_to_be_inf):
    for scenario in scenario_to_inf_metrics_dict.keys():

        #############################
        #values which are not relevant for either classification or regression scenarios will be set to None in scenarios_to_raw_res_dict:
        class_1_probs = scenario_to_raw_res_dict[scenario]['class_1_probs']

        reg_pred_intervals = scenario_to_raw_res_dict[scenario]['reg_pred_intervals']
        #############################

        predictions = scenario_to_raw_res_dict[scenario]['predictions']
        experi_vals = scenario_to_raw_res_dict[scenario]['experi_vals']

        for inf_metric in scenario_to_inf_metrics_dict[scenario]:
            expected_to_be_inf = False

            reasons_to_be_inf = metric_to_criteria_to_be_inf[inf_metric]

            for reason in reasons_to_be_inf:
                if zero_width_PIs == reason:
                    #####################
                    #This criterion is too strict. As long as the widths of the prediction intervals in a given bin were all zero, ENCE would become infinite!
                    if all([0==v for v in getIntervalsWidthOfInterest(sig_level_2_prediction_intervals={sig_level_of_interest:reg_pred_intervals},sig_level_of_interest=sig_level_of_interest).tolist()]):
                        expected_to_be_inf = True
                        print(f'Reason for inf metric = {reason} [metric = {inf_metric}, scenario = {scenario}')
                        break
                    ####################
          

            assert expected_to_be_inf,f'Unexpected inf metric {inf_metric}, for scenario = {scenario}'

def get_all_raw_res_pkls(top_dir_with_raw_res,dataset,regression_dataset_names,base_name_raw_res_pkl='RawResAndStats.pkl'):

    if not dataset in regression_dataset_names:
        all_raw_res_pkls = [os.path.sep.join([top_dir_with_raw_res,base_name_raw_res_pkl])]
    else:
        endpoint_specific_subdirs = reportSubDirs(top_dir_with_raw_res)
        all_raw_res_pkls = []
        for dir_ in endpoint_specific_subdirs:
            all_raw_res_pkls += [os.path.sep.join([dir_,base_name_raw_res_pkl])]


    return all_raw_res_pkls

def get_test_name_ignoring_fold_endpoint_prefix_fps_suffix_rnd_label_from_test_set_label(test_set_label,endpoint):

    test_name = test_set_label

    ###############
    #first step based on def get_test_name_ignoring_fold(...) from merge_exemplar_modelling_stats.py
    ###############

    test_name_ignoring_fold = re.sub('(_f\=[0-9]_test)','',test_name)
    ################

    test_name_ignoring_fold_endpoint_prefix = re.sub(f'({endpoint}_)','',test_name_ignoring_fold)

    test_name_ignoring_fold_endpoint_prefix_fp_suffix = re.sub('_FPs','',test_name_ignoring_fold_endpoint_prefix)

    final_test_name = test_name_ignoring_fold_endpoint_prefix_fp_suffix.replace('_RndTest0.2','')
    
    return final_test_name

def get_fold_from_test_set_label(test_set_label,type_of_modelling):
    ###################
    #based on def get_fold(...) from merge_exemplar_modelling_stats.py
    ##################

    if 'Classification' == type_of_modelling:
        fold = 'N/A'
    elif 'Regression' == type_of_modelling:
        try:
            fold = test_set_label.split('f=')[1].split('_')[0]
        except IndexError:
            raise Exception(f'Problems parsing this test_set_label={test_set_label}')
    else:
        raise Exception(f'Unrecognised type_of_modelling={type_of_modelling}')

    return fold


def construct_scenario_label(endpoint,test_name_ignoring_fold,fold,rand_seed,method,AD_method_name,AD_subset):
    scenario = f'e={endpoint}_t={test_name_ignoring_fold}_f={fold}_s={rand_seed}_alg={method}_ad={AD_method_name}_subset={AD_subset}'
    return scenario


def map_raw_res_onto_scenario_label(scenario_to_raw_res_default_dict_of_dict,contents_of_raw_res_pkl,dataset,regression_dataset_names):

    raw_res_dict = contents_of_raw_res_pkl['dict_of_raw_results']

    for endpoint in raw_res_dict.keys():
        for test_set_label in raw_res_dict[endpoint].keys():
            for rand_seed in raw_res_dict[endpoint][test_set_label].keys():
                for AD_method_name in raw_res_dict[endpoint][test_set_label][rand_seed].keys():
                    for method in raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name].keys():
                        for AD_subset in raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method].keys():
                            
                            test_name_ignoring_fold = get_test_name_ignoring_fold_endpoint_prefix_fps_suffix_rnd_label_from_test_set_label(test_set_label,endpoint)

                            #-----------------------------
                            if dataset in regression_dataset_names:
                                type_of_modelling = 'Regression'
                            else:
                                type_of_modelling = 'Classification'

                            fold = get_fold_from_test_set_label(test_set_label,type_of_modelling)
                            #------------------------------

                            #===================
                            #c.f. perform_classification_modelling_on_exemplar_datasets.py, perform_regression_modelling_on_exemplar_datasets.py, merge_exemplar_modelling_stats.py:
                            #AD_method_name = selected_ad_methods[0] = list(ADMethod2Param.keys())[0] when AD_subset = 'All', for raw results
                            #However, AD_method_name is then changed to 'N/A' for the 'All' subset when creating the file of combined statistics
                            #Cannot change key variable value directly: RuntimeError: dictionary changed size during iteration
                            #Moreover, would not want to change key variable value directly, as still need to use the original to extract raw results below!
                            if 'All' == AD_subset:
                                _AD_method_name = 'N/A' 
                            else:
                                _AD_method_name = AD_method_name
                            #===================

                            scenario = construct_scenario_label(endpoint,test_name_ignoring_fold,fold,rand_seed,method,_AD_method_name,AD_subset)

                            

                            #===================
                            #These should be unique across datasets!
                            assert not scenario in scenario_to_raw_res_default_dict_of_dict.keys(),scenario
                            #===================

                            if dataset in regression_dataset_names:
                                scenario_to_raw_res_default_dict_of_dict[scenario]['predictions'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_predictions'] 

                                scenario_to_raw_res_default_dict_of_dict[scenario]['experi_vals'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_y']

                                scenario_to_raw_res_default_dict_of_dict[scenario]['class_1_probs'] = None

                                scenario_to_raw_res_default_dict_of_dict[scenario]['reg_pred_intervals'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_sig_level_2_prediction_intervals'][sig_level_of_interest]
                            else:
                                scenario_to_raw_res_default_dict_of_dict[scenario]['predictions'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_predicted_y']

                                scenario_to_raw_res_default_dict_of_dict[scenario]['experi_vals'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_y']

                                scenario_to_raw_res_default_dict_of_dict[scenario]['class_1_probs'] = raw_res_dict[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_probs_for_class_1']

                                scenario_to_raw_res_default_dict_of_dict[scenario]['reg_pred_intervals'] = None
    
    return scenario_to_raw_res_default_dict_of_dict

           
def load_all_raw_results(top_class_or_reg_ds_dirs,regression_dataset_names):
    
    scenario_to_raw_res_default_dict_of_dict = defaultdict(dict)

    for dataset in top_class_or_reg_ds_dirs.keys():
        top_dir_with_raw_res = os.path.sep.join([top_class_or_reg_ds_dirs[dataset],'Modelling'])

        all_raw_res_pkls = get_all_raw_res_pkls(top_dir_with_raw_res,dataset,regression_dataset_names)

        for pkl in all_raw_res_pkls:
            contents_of_raw_res_pkl = load_from_pkl_file(pkl)

            scenario_to_raw_res_default_dict_of_dict = map_raw_res_onto_scenario_label(scenario_to_raw_res_default_dict_of_dict,contents_of_raw_res_pkl,dataset,regression_dataset_names)
    
    return scenario_to_raw_res_default_dict_of_dict

def identify_missing_or_inf_metrics(dict_of_merged_stats_files,regression_stats_in_desired_order,classification_stats_in_desired_order):
    scenario_to_missing_metrics_dict = defaultdict(list)

    scenario_to_inf_metrics_dict = defaultdict(list)

    #####################
    #Needed for checking as (c.f. perform_classification_modelling_on_exemplar_endpoints.py and perform_regression_modelling_on_exemplar_endpoints.py) raw results are not saved for zero compound subsets!)
    scenario_to_cmpd_counts_dict = {}
    ####################

    for dataset_type in dict_of_merged_stats_files.keys():
        if 'Classification' == dataset_type:
            stats_names = classification_stats_in_desired_order
        elif 'Regression' == dataset_type:
            stats_names = regression_stats_in_desired_order
        else:
            raise Exception(f'Unrecognised dataset_type={dataset_type}')
        
        df = pd.read_csv(dict_of_merged_stats_files[dataset_type])

        records = df.to_dict(orient='records')

        for row in records:
            endpoint = row['Endpoint']
            test_name_ignoring_fold = row['Test Set Name (ignoring fold if applicable)']
            fold = row['Fold (if applicable)']
            if pd.isna(fold):
                fold = 'N/A'
            rand_seed = row['Random seed']
            method = row['Modelling Algorithm']
            AD_method_name = row['AD Method']
            if pd.isna(AD_method_name):
                AD_method_name = 'N/A'
            AD_subset = row['AD Subset']

            scenario = construct_scenario_label(endpoint,test_name_ignoring_fold,fold,rand_seed,method,AD_method_name,AD_subset)

            for metric in stats_names:
                if pd.isna(row[metric]):
                    scenario_to_missing_metrics_dict[scenario].append(metric)
                    print(f'Found a missing metric: metric = {metric}, scenario = {scenario}')
                
                if np.isinf(row[metric]):
                    scenario_to_inf_metrics_dict[scenario].append(metric)
                    print(f'Found a inf metric: metric = {metric}, scenario = {scenario}')
                    assert 'ENCE' == metric,metric
                    assert row[metric] > 0,row[metric]
            
            scenario_to_cmpd_counts_dict[scenario] = row['no. compounds']
    
    return scenario_to_missing_metrics_dict,scenario_to_inf_metrics_dict,scenario_to_cmpd_counts_dict

def main():
    print('THE START')

    ##############
    #scenarios need to be defined consistently and each one should correspond to a unique combination of endpoint (unique across datasets), test set name, fold, seed, modelling algorithm, AD method, AD Subset
    scenario_to_raw_res_dict = load_all_raw_results(top_class_or_reg_ds_dirs,regression_dataset_names) 

    scenario_to_missing_metrics_dict,scenario_to_inf_metrics_dict,scenario_to_cmpd_counts_dict = identify_missing_or_inf_metrics(dict_of_merged_stats_files,regression_stats_in_desired_order,classification_stats_in_desired_order)
    ###############

    check_missing_metrics(scenario_to_missing_metrics_dict,scenario_to_raw_res_dict,metric_to_criteria_to_be_missing,scenario_to_cmpd_counts_dict)

    check_inf_metrics(scenario_to_inf_metrics_dict,scenario_to_raw_res_dict,metric_to_criteria_to_be_inf)

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
