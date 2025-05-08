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
#Copright (c) 2022-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Contact zied.hosni [at] syngenta.com
###############################
import os,   sys,   time,   pickle, shutil, itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
#----------------------------------
from common_globals_regression_scripts import updateDictOfRawInsideVsOutsideADResults
#---------------------------------
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
public_data_scripts_general_purpose_dir = os.path.sep.join([os.path.dirname(dir_of_this_script), 'general_purpose'])
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#---------------------------------
sys.path.append(public_data_scripts_general_purpose_dir)
#---------------------------------
##################################
from common_globals import consistent_id_col,   consistent_act_col,   consistent_smiles_col,   consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,   top_class_or_reg_ds_dirs,   regression_dataset_names,   classification_dataset_names,   ds_matched_to_ep_list,   get_endpoint_type
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,   fp_col,   stand_mol_inchi_col,   stand_mol_smiles_col,   subset_name_col
from common_globals import get_rand_subset_train_test_suffix,   rand_split_seed,   no_train_test_splits
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
from common_globals import ds_matched_to_exemplar_ep_list
from common_globals import load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations
from common_globals import all_global_random_seed_opts
from common_globals import class_1,   class_0
from common_globals import rnd_test_suffix,  wang_test_set_names
from common_globals import class_ds_name_to_test_subset_names
from common_globals import populateDictWithStats, writeStatsFiles
##################################
from define_plot_settings import *
##################################
#---------------------------------
sys.path.append(top_scripts_dir)
#---------------------------------
from consistent_parameters_for_all_modelling_runs import ADMethod2Param, nrTrees_regression, native_uncertainty_alg_variant, ml_alg_regression, non_conformity_scaling, icp_calib_fraction, number_of_acp_splits, number_of_scp_splits, sig_level_of_interest, all_sig_levels_considered_to_compute_ECE
from consistent_parameters_for_all_modelling_runs import consistent_k
ml_alg = ml_alg_regression
nrTrees = nrTrees_regression
#---------------------------------
sys.path.append(pkg_dir)
#---------------------------------
from modelsADuncertaintyPkg.qsar_class_uncertainty import define_native_ML_baseline as nml
from modelsADuncertaintyPkg.qsar_class_uncertainty import IVAP_CVAP_workflow_functions as vap
from modelsADuncertaintyPkg.qsar_eval import all_key_class_stats_and_plots as ClassEval
from modelsADuncertaintyPkg.qsar_class_uncertainty.parse_native_or_venn_abers_output import getProbsForClass1
from modelsADuncertaintyPkg.utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD
from modelsADuncertaintyPkg.utils.ML_utils import getXandY, singleRandomSplit, makeYLabelsNumeric
from modelsADuncertaintyPkg.qsar_class_uncertainty import define_native_ML_baseline as nml
from modelsADuncertaintyPkg.utils.ML_utils import predictedBinaryClassFromProbClass1
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import findDups, convertDefaultDictDictIntoDataFrame, neverEndingDefaultDict, findDups
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds, getInsideOutsideADSubsets
from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import getFpsDfAndDataDfSkippingFailedMols
from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,  convertDefaultDictDictIntoDataFrame,  neverEndingDefaultDict,  findDups
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,  makeIDsNumeric,  prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD,  getADSubsetTestIDsInOrder
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,  create_pkl_file
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.qsar_reg_uncertainty import funcsToApplyRegUncertaintyToNewDatasets as RegUncert
from modelsADuncertaintyPkg.qsar_eval import all_key_reg_stats_and_plots as RegEval
#=================================
selected_uncertainty_methods = ['Native', 'ICP', 'ACP', 'SCP']
selected_ad_methods = list(ADMethod2Param.keys())
#=================================



def get_nested_dicts_of_modelling_results_for_all_exemplar_endpoints(endpoints_of_interest,dataset_name, dict_of_stats, statsFilesDict, dict_of_raw_results, ds_matched_to_exemplar_ep_list, selected_uncertainty_methods, selected_ad_methods, all_global_random_seed_opts, consistent_id_col,   consistent_act_col, dir_with_files_to_parse, top_calc_dir,ADMethod2Param=ADMethod2Param,default_k=consistent_k,default_Tanimoto_thresh=ADMethod2Param['Tanimoto']):
    
    for endpoint in endpoints_of_interest:
        print(f'endpoint={endpoint}')

        if not endpoint in ds_matched_to_exemplar_ep_list[dataset_name]: raise Exception(f'Unexpected endpoint = {endpoint}')

        #=============================
        #if (not ADMethod2Param['Tanimoto'] == default_Tanimoto_thresh) or (not ADMethod2Param['UNC'] == default_k):
        short_res_dir_name = f"{endpoint}_k={ADMethod2Param['UNC']}_t={ADMethod2Param['Tanimoto']}"
        #else:
        #    short_res_dir_name = endpoint
        #=============================
        
        calc_dir = os.path.sep.join([top_calc_dir,short_res_dir_name])

        createOrReplaceDir(dir_=calc_dir)

        regularly_updated_pkl_file_with_stats_and_raw_results = os.path.sep.join([calc_dir, 'RawResAndStats.pkl'])
        
        
        
        
        for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
            
            train_set_label = train_test_label_pair[0]
            test_set_label = train_test_label_pair[1]
            
            fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids =load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,  train_set_label,  test_set_label)
            
            #--------------------
            orig_test_ids = X_test_and_ids_df[consistent_id_col].tolist()
            orig_test_ids.sort()
            test_ids.sort()
            assert test_ids==orig_test_ids
            del test_ids
            #--------------------
            
            X_train_and_ids_df = makeIDsNumeric(df=X_train_and_ids_df,  id_col=consistent_id_col)
            
            X_test_and_ids_df = makeIDsNumeric(df=X_test_and_ids_df,  id_col=consistent_id_col)
            
            test_ids = X_test_and_ids_df[consistent_id_col].tolist() #This is needed for (at least) d1NN ("Tanimoto") AD method.
            
            train_inc_calib_x, train_inc_calib_y, test_x = prepareInputsForModellingAndUncertainty(X_train_and_ids_df, train_y, X_test_and_ids_df, id_col=consistent_id_col)
            
            #==================
            #Redundant:
            del train_y
            #==================
            
            for rand_seed in all_global_random_seed_opts:
                #----------------------
                global_random_seed = rand_seed
                #----------------------
                for uncertainty_method in selected_uncertainty_methods:
                    #--------------------------
                    method = uncertainty_method
                    #--------------------------
                    
                    sig_level_2_test_predictions,sig_level_2_prediction_intervals = RegUncert.getTestPredictionsAndPredictionIntervalsForAllSignificanceLevels(all_sig_levels_considered_to_compute_ECE, uncertainty_method, train_inc_calib_x, train_inc_calib_y, test_x, non_conformity_scaling, ml_alg, global_random_seed, number_of_scp_splits, number_of_acp_splits, icp_calib_fraction, nrTrees, native_uncertainty_alg_variant,allow_for_acp_predictions_could_be_inconsistent=True)
                    
                    test_predictions = sig_level_2_test_predictions[sig_level_of_interest] #ACP can return predictions which depend upon the prediction intervals!

                    for AD_method_name in selected_ad_methods:
                        
                        context_label = f't={test_set_label}_s={rand_seed}_ad={AD_method_name}_m={method}'
                        
                        test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df, X_test=X_test_and_ids_df, fps_train=fps_train, fps_test=fps_test, threshold=ADMethod2Param[AD_method_name], id_col=consistent_id_col, rand_seed=rand_seed, endpoint_col=consistent_act_col, AD_method_name=AD_method_name, test_ids=test_ids, y_train=train_inc_calib_y, y_test=None, regression=True, class_1_label=None, class_0_label=None)
                        
                        if AD_method_name == selected_ad_methods[0]:
                            ad_subsets_to_consider = ['All', 'Inside', 'Outside']
                        else:
                            ad_subsets_to_consider = ['Inside', 'Outside']
                        
                        for AD_subset in ad_subsets_to_consider:
                            
                            subset_test_ids = getADSubsetTestIDsInOrder(AD_subset, test_id_status_dict, test_ids)
                            
                            if not 0 == len(subset_test_ids):
                                
                                #===============================================
                                subset_test_y = getSubsetYFromSubsetIds(test_ids, pd.Series(test_y), subset_test_ids).tolist()
                                
                                subset_test_predictions = getSubsetYFromSubsetIds(test_ids, pd.Series(test_predictions), subset_test_ids).tolist()
                                
                                subset_sig_level_2_prediction_intervals = RegEval.getSigLevel2SubsetPredictionIntervals(test_ids, sig_level_2_prediction_intervals, subset_test_ids)
                                
                                #===============================================
                                
                                #-----------------------------------------------
                                updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_test_predictions,subset_sig_level_2_prediction_intervals)
                                #-----------------------------------------------
                                
                                rmse, MAD, R2, Pearson, Pearson_Pval_one_tail, Spearman, Spearman_Pval_one_tail, validity, efficiency, ECE_new, ENCE, errorRate_s, no_compounds,scc = RegEval.computeAllRegMetrics(test_y=subset_test_y, test_predictions=subset_test_predictions, sig_level_of_interest=sig_level_of_interest, sig_level_2_prediction_intervals=subset_sig_level_2_prediction_intervals)
                                
                                RegEval.plotPredPlusEstimatedErrorBarsVsExperi(y_test=subset_test_y, testPred=subset_test_predictions, sig_level_2_prediction_intervals=subset_sig_level_2_prediction_intervals, sig_level_of_interest=sig_level_of_interest, plot_file_name=os.path.sep.join([calc_dir, f'{context_label}_{AD_subset}_Reg_Experi_vs_Pred_Plus_Err_Bars.tiff']), plot_title='')
                                
                            else:
                                rmse, MAD, R2, Pearson, Pearson_Pval_one_tail, Spearman, Spearman_Pval_one_tail, validity, efficiency, ECE_new, ENCE, errorRate_s, scc = [None]*13
                                no_compounds = 0
                
                            del errorRate_s
                            
                            dict_of_current_stats = RegEval.map_all_regression_stats_onto_default_names(rmse, MAD, R2, Pearson, Pearson_Pval_one_tail, Spearman, Spearman_Pval_one_tail, validity, efficiency, ECE_new, ENCE, no_compounds,scc)
                                
                            populateDictWithStats(dict_of_stats,dict_of_current_stats,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset)
                                
                            statsFilesDict = writeStatsFiles(statsFilesDict,dataset_name,dict_of_stats,calc_dir,context_label,endpoint,test_set_label,rand_seed,AD_method_name,method)
                                
                            create_pkl_file(regularly_updated_pkl_file_with_stats_and_raw_results, {'dict_of_raw_results':dict_of_raw_results, 'dict_of_stats':dict_of_stats, 'statsFilesDict':statsFilesDict})
    
    return dict_of_raw_results, dict_of_stats, statsFilesDict


def main():
    print('THE START')

    #----------------------
    assert 1 == len(regression_dataset_names)
    dataset_name = regression_dataset_names[0]
    #----------------------

    #####################################
    parser = argparse.ArgumentParser(
        description='Specify period (".") list of endpoints to generate results for in case running all endpoints at once takes too long - as well as non-default AD method parameters.')
    parser.add_argument('-e', dest="endpoints_of_interest_str", action='store', type=str,
                        help='Specify period (".") list of endpoints to generate results for', default='.'.join(ds_matched_to_exemplar_ep_list[dataset_name]))
    parser.add_argument('-k',dest="k_for_AD_methods",action='store',type=int,default=consistent_k)
    parser.add_argument('-t',dest="tanimoto_distance_threshold",action='store',type=float,default=ADMethod2Param['Tanimoto'])
    ##############################
    ##############################
    dict_of_opts_which_can_be_set_from_cmdline = vars(parser.parse_args())
    print('*'*50)
    print('Running calculations with the following options specified:')
    for var_name in dict_of_opts_which_can_be_set_from_cmdline.keys():
        print(var_name, '=',
              dict_of_opts_which_can_be_set_from_cmdline[var_name])
    print('*'*50)
    #############################
    endpoints_of_interest = dict_of_opts_which_can_be_set_from_cmdline['endpoints_of_interest_str'].split('.')
    k_for_AD_methods = dict_of_opts_which_can_be_set_from_cmdline['k_for_AD_methods']
    tanimoto_distance_threshold = dict_of_opts_which_can_be_set_from_cmdline['tanimoto_distance_threshold']

    if (not k_for_AD_methods == consistent_k) or (not tanimoto_distance_threshold == ADMethod2Param['Tanimoto']):
        ADMethod2Param_to_be_used = {}
        ADMethod2Param_to_be_used['Tanimoto'] = tanimoto_distance_threshold
        ADMethod2Param_to_be_used['dkNN'] = k_for_AD_methods
        ADMethod2Param_to_be_used['RDN'] = k_for_AD_methods
        ADMethod2Param_to_be_used['UNC'] = k_for_AD_methods

    else:
        ADMethod2Param_to_be_used = ADMethod2Param
    #############################
    
    for dataset_name in regression_dataset_names:
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        top_calc_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Modelling'])
        
        assert not top_calc_dir == dir_with_files_to_parse,   dir_with_files_to_parse

        #createOrReplaceDir(top_calc_dir)

        
        dict_of_raw_results = neverEndingDefaultDict()
        
        dict_of_stats = neverEndingDefaultDict()
        
        statsFilesDict = neverEndingDefaultDict()
        
        
        dict_of_raw_results, dict_of_stats, statsFilesDict = get_nested_dicts_of_modelling_results_for_all_exemplar_endpoints(endpoints_of_interest,dataset_name, dict_of_stats, statsFilesDict, dict_of_raw_results, ds_matched_to_exemplar_ep_list, selected_uncertainty_methods, selected_ad_methods, all_global_random_seed_opts, consistent_id_col,   consistent_act_col, dir_with_files_to_parse, top_calc_dir,ADMethod2Param=ADMethod2Param_to_be_used)
    
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
