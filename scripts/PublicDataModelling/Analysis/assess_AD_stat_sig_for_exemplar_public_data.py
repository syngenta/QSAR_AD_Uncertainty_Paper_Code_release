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
#Script co-written by Zied Hosni
###############################
import os,sys,time,pickle,re
import csv
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
top_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
sys.path.append(top_dir)
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from consistent_parameters_for_all_modelling_runs import ADMethod2Param
from consistent_parameters_for_all_modelling_runs import consistent_no_random_splits_for_AD_p_val
from PublicDataModelling.general_purpose.extract_ad_params_from_abs_file_name import get_ad_params
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
data_dir = os.path.dirname(os.path.dirname(pkg_dir))
#raise Exception(f'top_dir containing PublicData = {data_dir}')
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_eval.assess_stat_sig_shift_metrics import workflowForAFamilyOfRawResults, estimated_probabilities_are_too_close
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
import pickle
import pandas as pd
import os
import pandas as pd
import os
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
from modelsADuncertaintyPkg.qsar_eval.all_key_reg_stats_and_plots import getIntervalsWidthOfInterest

def load(fn):
    data = None
    file = open(fn, "rb")
    try:
        data = pickle.load(file)
        print(data.keys())
        
        for key, values in data.items():
            print(f"Key: {key}, Values Type: {type(values)}")  
            print(f"Key: {key}, Values Content: {values}")    
           
    finally:
        file.close()
    return data

def load_the_pickled_data(dataset_name, data_dir,ADMethod2Param):
    if dataset_name in ["Tox21", "Morger_ChEMBL"]:
        folder_path = os.path.sep.join([data_dir, "PublicData", dataset_name, "Modelling"])
        type_of_modelling = 'binary_class'
    elif dataset_name == "Wang_ChEMBL":
        folder_path = os.path.sep.join([data_dir,"PublicData",dataset_name,"data","dataset","Modelling"])
        type_of_modelling = "regression"
    else:
        raise Exception(f"Unrecognised dataset name={dataset_name}")

    info = combine_pkl_files(folder_path,ADMethod2Param)

    #print(info.keys())

    return(info, folder_path, type_of_modelling)

def extract_raw_results(original_dictionary_of_raw_data):
    dict_for_all_test_sets_of_raw_subset_results = original_dictionary_of_raw_data['dict_of_raw_results']
    return(dict_for_all_test_sets_of_raw_subset_results)


def reshape_dict(old_dict,dataset_name):
    
    new_dict = {}

    if not dataset_name == "Wang_ChEMBL":
        # Iterate over the endpoints
        for endpoint, tests in old_dict.items():
            # Iterate over the test configurations
            for test_config, seeds in tests.items():
                # Iterate over the seeds
                for seed, ads in seeds.items():
                    # Iterate over the ADs
                    for ad, algs in ads.items():
                        # Iterate over the Algorithms
                        for alg, data in algs.items():
                            # Construct the new key
                            new_key = f"{endpoint}_{test_config.split('_')[1]}_test_FPs_{ad}_{alg}"
                            new_seed = f"f=0_{seed}"

                            # Create nested dictionaries if they don't exist
                            if new_key not in new_dict:
                                new_dict[new_key] = {}
                            if new_seed not in new_dict[new_key]:
                                new_dict[new_key][new_seed] = {}

                            # Assign the data to the new key structure
                            new_dict[new_key][new_seed]["Inside"] = data["Inside"]
                            new_dict[new_key][new_seed]["Outside"] = data["Outside"]
    else:
        # Iterate over the endpoints
        for endpoint, tests in old_dict.items():
            # Iterate over the test configurations
            for test_config, seeds in tests.items():
                # Extract the fold number and remove it from the test_config
                fold = test_config.split('_f=')[1].split('_')[0]
                # Remove the endpoint from the test_config
                test_config_base = test_config.replace(f"{endpoint}_", '').replace(f'_f={fold}', '')

                # Iterate over the seeds
                for seed, ads in seeds.items():
                    # Iterate over the ADs
                    for ad, algs in ads.items():
                        # Iterate over the Algorithms
                        for alg, data in algs.items():
                            # Construct the new key
                            new_key = f"{endpoint}_{test_config_base}_{ad}_{alg}"
                            new_fold_seed = f"f={fold}_{seed}"

                            # Create nested dictionaries if they don't exist
                            if new_key not in new_dict:
                                new_dict[new_key] = {}
                            if new_fold_seed not in new_dict[new_key]:
                                new_dict[new_key][new_fold_seed] = {}

                            # Assign the data to the new key structure
                            new_dict[new_key][new_fold_seed] = data
    return new_dict

def remove_key_na(d):
    keys_to_delete = []
    for key, value in d.items():
        if key == "All":
            keys_to_delete.append(key)
        elif isinstance(value, dict):
            d[key] = remove_key_na(value)

    for key in keys_to_delete:
        del d[key]

    return d

def add_dummy_values_for_missing_subset_results(context_mapped_to_dict_for_p_value_calculation,dataset_name):
    #For subsets with too few compounds, no modelling results will have been generated, but this could cause problems with downstream calculations!
    #So, just treat these as corresponding to empty lists of experimental and predicted values!
    expected_subsets = ['Inside','Outside']
    for context in context_mapped_to_dict_for_p_value_calculation.keys():
        for fold_seed_combination in context_mapped_to_dict_for_p_value_calculation[context].keys():
            for subset in expected_subsets:
                if 0 == len(context_mapped_to_dict_for_p_value_calculation[context][fold_seed_combination][subset]) or not subset in context_mapped_to_dict_for_p_value_calculation[context][fold_seed_combination].keys():
                    print(f'Adding dummy values for missing results for subset={subset}')
                    context_mapped_to_dict_for_p_value_calculation[context][fold_seed_combination][subset]['subset_test_y'] = []
                    
                    if "Wang_ChEMBL" == dataset_name:
                        context_mapped_to_dict_for_p_value_calculation[context][fold_seed_combination][subset]['subset_test_predictions'] = []
                    
                    elif dataset_name in ["Morger_ChEMBL","Tox21"]:
                        context_mapped_to_dict_for_p_value_calculation[context][fold_seed_combination][subset]['subset_predicted_y'] = []
                    
                    else:
                        raise Exception(f'Unrecognised dataset = {dataset_name}')

    return context_mapped_to_dict_for_p_value_calculation




def prepare_raw_data_for_pValue_calculation(dict_for_all_test_sets_of_raw_subset_results,dataset_name):
    context_mapped_to_dict_for_p_value_calculation = reshape_dict(dict_for_all_test_sets_of_raw_subset_results, dataset_name=dataset_name)
    context_mapped_to_dict_for_p_value_calculation = remove_key_na(context_mapped_to_dict_for_p_value_calculation)
    context_mapped_to_dict_for_p_value_calculation = add_dummy_values_for_missing_subset_results(context_mapped_to_dict_for_p_value_calculation,dataset_name)
       
    return context_mapped_to_dict_for_p_value_calculation


def compute_p_values_for_all_context(context_mapped_to_dict_for_p_value_calculation,out_dir,
                                     subset_1_name,subset_2_name,type_of_modelling,no_rand_splits,metrics,
                                     metric_to_expected_sub_1_minus_sub_2_sign, results_are_obtained_over_multiple_folds_and_or_seeds=True,
                                     debug=True, scenarios_with_errors=[],set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True):
    
        
    workflowForAFamilyOfRawResults(out_dir, context_mapped_to_dict_for_p_value_calculation,
                                                                subset_1_name=subset_1_name,
                                   subset_2_name=subset_2_name, type_of_modelling=type_of_modelling, no_rand_splits=no_rand_splits,
                                   strat_rand_split_y_name='strat_y', rand_seed=42, metrics_of_interest=metrics, x_lab='Groups',
                                   y_lab='Shift-Metric', legend_lab='Split Basis', sig_level_perc=5, one_sided_sig_test=True,
                                   metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,
                                   p_vals_table='one_tail_{}_PVals.csv'.format(type_of_modelling), create_plots=True,
                                                                results_are_obtained_over_multiple_folds_and_or_seeds=results_are_obtained_over_multiple_folds_and_or_seeds, debug=debug, scenarios_with_errors=scenarios_with_errors,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=set_p_to_nan_if_metric_cannot_be_computed_for_orig_split)

    workflowForAFamilyOfRawResults(out_dir, context_mapped_to_dict_for_p_value_calculation,
                                                                subset_1_name=subset_1_name,
                                   subset_2_name=subset_2_name, type_of_modelling=type_of_modelling, no_rand_splits=no_rand_splits,
                                   strat_rand_split_y_name='strat_y', rand_seed=42, metrics_of_interest=metrics, x_lab='Groups',
                                   y_lab='Shift-Metric', legend_lab='Split Basis', sig_level_perc=5, one_sided_sig_test=False,
                                   metric_to_expected_sub_1_minus_sub_2_sign=metric_to_expected_sub_1_minus_sub_2_sign,
                                   p_vals_table='{}_PVals.csv'.format(type_of_modelling), create_plots=True,
                                                                results_are_obtained_over_multiple_folds_and_or_seeds=results_are_obtained_over_multiple_folds_and_or_seeds, debug=debug, scenarios_with_errors=scenarios_with_errors,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=set_p_to_nan_if_metric_cannot_be_computed_for_orig_split)


def file_corresponds_to_default_ad_method_parameters(pkl_file,ADMethod2Param):
    default_k = ADMethod2Param["UNC"]
    default_distance_threshold = ADMethod2Param["Tanimoto"]

    k_val,tanimoto_distance_threshold = get_ad_params(pkl_file)

    if k_val == default_k and tanimoto_distance_threshold == default_distance_threshold:
        return True
    else:
        return False



def combine_pkl_files(folder_path,ADMethod2Param):
    combined_dict = {'dict_of_stats': {}, 'dict_of_raw_results': {}, 'statsFilesDict': {}}
    for folder_contents_tuple in os.walk(folder_path):
        top_dir, list_of_top_dir_and_dirs, files_in_any_of_these_dirs = folder_contents_tuple
        root = top_dir
        dirs = list_of_top_dir_and_dirs
        files = files_in_any_of_these_dirs
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)

                if not file_corresponds_to_default_ad_method_parameters(pkl_file=file_path,ADMethod2Param=ADMethod2Param):
                    print(f'Skipping {file_path}')
                    continue

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    for key in ['dict_of_stats', 'dict_of_raw_results', 'statsFilesDict']:
                        if key in data:
                            for sub_key, value in data[key].items():
                                # Merge contents for keys with the same title
                                if sub_key in combined_dict[key]:
                                    combined_dict[key][sub_key].update(value)
                                else:
                                    combined_dict[key][sub_key] = value
    return combined_dict


def main():
    print('THE START')
    ########################
    subset_1_name='Inside'
    subset_2_name='Outside'
    no_rand_splits=consistent_no_random_splits_for_AD_p_val
    all_datasets = ["Wang_ChEMBL","Morger_ChEMBL","Tox21"]
    
    debug = True
    scenarios_with_errors = []
    ########################
    for dataset_name in all_datasets:

        original_dictionary_of_raw_data, out_dir,type_of_modelling = load_the_pickled_data(dataset_name, data_dir,ADMethod2Param)
        
        dict_for_all_test_sets_of_raw_subset_results = extract_raw_results(original_dictionary_of_raw_data)
        context_mapped_to_dict_for_p_value_calculation = prepare_raw_data_for_pValue_calculation(dict_for_all_test_sets_of_raw_subset_results,
                                                                                               dataset_name)
        ####################################
        #Debug:
        #if not 'Tox21' == dataset_name:
        #    continue
        #else:
        #    print(f'context_mapped_to_dict_for_p_value_calculation["NR-Aromatase_Tox21Score_test_FPs_dkNN_Native"]={context_mapped_to_dict_for_p_value_calculation["NR-Aromatase_Tox21Score_test_FPs_dkNN_Native"]}')
        #    sys.exit()
        ###################################
        
        if "Wang_ChEMBL" == dataset_name:
            metric_to_expected_sub_1_minus_sub_2_sign = regression_metric_to_expected_sub_1_minus_sub_2_sign
        elif dataset_name in ["Morger_ChEMBL", "Tox21"]:
            metric_to_expected_sub_1_minus_sub_2_sign = classifcation_metric_to_expected_sub_1_minus_sub_2_sign
        else:
            raise Exception(f"{dataset_name} is not a valid name.")
        
        metrics = list(metric_to_expected_sub_1_minus_sub_2_sign.keys())
        
        compute_p_values_for_all_context(context_mapped_to_dict_for_p_value_calculation,out_dir,
                                     subset_1_name,subset_2_name,type_of_modelling,no_rand_splits,metrics,
                                     metric_to_expected_sub_1_minus_sub_2_sign, results_are_obtained_over_multiple_folds_and_or_seeds=True, debug=debug, scenarios_with_errors=scenarios_with_errors,set_p_to_nan_if_metric_cannot_be_computed_for_orig_split=True)
        
    print('THE END')
    return 0

if __name__ == '__main__':
    sys.exit(main())
