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
from common_globals_classification_scripts import updateDictOfRawInsideVsOutsideADResults
#---------------------------------
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
public_data_scripts_general_purpose_dir = os.path.sep.join([os.path.dirname(dir_of_this_script),'general_purpose'])
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
from common_globals import populateDictWithStats,writeStatsFiles
##################################
from define_plot_settings import *
##################################
#---------------------------------
sys.path.append(top_scripts_dir)
#---------------------------------
from consistent_parameters_for_all_modelling_runs import ADMethod2Param, ml_alg_classification,lit_precedent_delta_for_calib_plot,larger_delta_for_calib_plot
from consistent_parameters_for_all_modelling_runs import consistent_k
ml_alg = ml_alg_classification
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
#=================================
selected_uncertainty_methods = ['Native', 'IVAP', 'CVAP']
selected_ad_methods = list(ADMethod2Param.keys())
#=================================



def get_nested_dicts_of_modelling_results_for_all_exemplar_endpoints(dataset_name, dict_of_stats, statsFilesDict, dict_of_raw_results, ds_matched_to_exemplar_ep_list, selected_uncertainty_methods, selected_ad_methods, all_global_random_seed_opts, consistent_id_col,   consistent_act_col, dir_with_files_to_parse, calc_dir, class_1, class_0,ADMethod2Param=ADMethod2Param):
    
    regularly_updated_pkl_file_with_stats_and_raw_results = os.path.sep.join([calc_dir, 'RawResAndStats.pkl'])
    
    for endpoint in ds_matched_to_exemplar_ep_list[dataset_name]:
        print(f'endpoint={endpoint}')
        
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
            
            ########################
            #This should not be needed for public classification datasets:
            #train_inc_calib_y = makeYLabelsNumeric(train_inc_calib_y, class_1=class_1, class_0=class_0)
            #test_y = makeYLabelsNumeric(test_y, class_1=class_1, class_0=class_0)
            #######################
    
            #==================
            #Redundant:
            del train_y
            #==================
            
            if 'IVAP' in selected_uncertainty_methods:
                run_ivap = True
            else:
                run_ivap = False
            
            if 'CVAP' in selected_uncertainty_methods:
                run_cvap = True
            else:
                run_cvap = False
            
            for rand_seed in all_global_random_seed_opts:
            
                IVAP_calib_score_label_tuples, IVAP_model, IVAP_test_id_to_pred_class_class_1_prob_p1_p0, CVAP_calib_score_label_tuples_per_fold, CVAP_models_per_fold, CVAP_test_id_to_pred_class_class_1_prob_p1_p0 = vap.runIVAPandOrCVAPWorkflow(train_inc_calib_x, train_inc_calib_y, test_x, train_ids, test_ids, ml_alg, rand_seed, class_1, test_y,run_cvap=run_cvap,run_ivap=run_ivap)
                
                if 'Native' in selected_uncertainty_methods:
                    model, native_test_id_to_pred_class_class_1_prob = nml.getNativeModelAndTestId2PredClassAndClass1Prob(train_inc_calib_x, train_inc_calib_y, ml_alg, rand_seed, class_1, test_x, test_ids)
                    
                    del model
                else:
                    native_test_id_to_pred_class_class_1_prob = None
                
                for AD_method_name in selected_ad_methods:
                
                    test_id_status_dict = findInsideOutsideADTestIds(X_train=X_train_and_ids_df, X_test=X_test_and_ids_df, fps_train=fps_train, fps_test=fps_test, threshold=ADMethod2Param[AD_method_name], id_col=consistent_id_col,rand_seed=rand_seed, endpoint_col=consistent_act_col, AD_method_name=AD_method_name,test_ids=test_ids, y_train=train_inc_calib_y, y_test=None, regression=False,class_1_label=class_1,class_0_label=class_0)
                    
                    if AD_method_name == selected_ad_methods[0]:
                        ad_subsets_to_consider = ['All','Inside','Outside']
                    else:
                        ad_subsets_to_consider = ['Inside','Outside']
                    
                    for AD_subset in ad_subsets_to_consider:
                        
                        subset_test_ids = getADSubsetTestIDsInOrder(AD_subset,test_id_status_dict,test_ids)
                        
                        for method in selected_uncertainty_methods:
                            context_label = f't={test_set_label}_s={rand_seed}_ad={AD_method_name}_m={method}'
                            
                            if not 0 == len(subset_test_ids):
                                #==================================================
                                subset_test_y = getSubsetYFromSubsetIds(test_ids,pd.Series(test_y),subset_test_ids).tolist()
                                
                                
                                subset_probs_for_class_1 = getProbsForClass1(method,IVAP_test_id_to_pred_class_class_1_prob_p1_p0,CVAP_test_id_to_pred_class_class_1_prob_p1_p0,native_test_id_to_pred_class_class_1_prob,subset_test_ids)
                                
                                subset_predicted_y = [predictedBinaryClassFromProbClass1(p) for p in subset_probs_for_class_1]
                                #===================================================
                                
                                updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_probs_for_class_1,subset_predicted_y)
                                
                                #----------------------------------------------------
                                
                                precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds =        ClassEval.computeAllClassMetrics(test_y=subset_test_y,predicted_y=subset_predicted_y,probs_for_class_1=subset_probs_for_class_1,method=method,subset_name=f'{context_label}_{AD_subset}',output_dir=calc_dir,delta_for_calib_plot=lit_precedent_delta_for_calib_plot)
                                
                                ClassEval.computeAllClassMetrics(test_y=subset_test_y,predicted_y=subset_predicted_y,probs_for_class_1=subset_probs_for_class_1,method=method,subset_name=f'{context_label}_{AD_subset}',output_dir=calc_dir,delta_for_calib_plot=larger_delta_for_calib_plot)
                            else:
                            
                                precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal = [None]*17
                                no_cmpds_1,no_cmpds_0,no_cmpds = [0]*3
                            
                            dict_of_current_stats = ClassEval.map_all_class_stats_onto_default_names(precision_1,precision_0,recall_1,recall_0,ba,mcc,auc,kappa,brier,strat_brier,rmseCal, MADCal, coeffOfDeterminationCal, PearsonCoeffCal, PearsonCoeffPvalCal, SpearmanCoeffCal,SpearmanCoeffPvalCal,no_cmpds_1,no_cmpds_0,no_cmpds)
                            
                            populateDictWithStats(dict_of_stats,dict_of_current_stats,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset)
                            
                            statsFilesDict = writeStatsFiles(statsFilesDict,dataset_name,dict_of_stats,calc_dir,context_label,endpoint,test_set_label,rand_seed,AD_method_name,method)
                            
                            create_pkl_file(regularly_updated_pkl_file_with_stats_and_raw_results,{'dict_of_raw_results':dict_of_raw_results,'dict_of_stats':dict_of_stats,'statsFilesDict':statsFilesDict})
    
    return dict_of_raw_results, dict_of_stats, statsFilesDict


def main():
    print('THE START')

    #####################################
    parser = argparse.ArgumentParser(
        description='Specify non-default AD method parameters.')
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
    
    for dataset_name in classification_dataset_names:
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        top_calc_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   f"Modelling"])
        
        
        assert not top_calc_dir == dir_with_files_to_parse,   dir_with_files_to_parse
        
        createOrReplaceDir(dir_=top_calc_dir)

        calc_dir = os.path.sep.join([top_calc_dir,f"k={ADMethod2Param_to_be_used['UNC']}_t={ADMethod2Param_to_be_used['Tanimoto']}"])

        createOrReplaceDir(dir_=calc_dir)
        
        dict_of_raw_results = neverEndingDefaultDict()
        
        dict_of_stats = neverEndingDefaultDict()
        
        statsFilesDict = neverEndingDefaultDict()
        
        
        dict_of_raw_results, dict_of_stats, statsFilesDict = get_nested_dicts_of_modelling_results_for_all_exemplar_endpoints(dataset_name, dict_of_stats, statsFilesDict, dict_of_raw_results, ds_matched_to_exemplar_ep_list, selected_uncertainty_methods, selected_ad_methods, all_global_random_seed_opts, consistent_id_col,   consistent_act_col, dir_with_files_to_parse, calc_dir, class_1, class_0,ADMethod2Param=ADMethod2Param_to_be_used)
    
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
