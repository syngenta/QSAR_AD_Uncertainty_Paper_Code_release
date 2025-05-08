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
#Copright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import os,sys,time,pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import shutil
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_scripts_dir = os.path.dirname(dir_of_this_script)
pkg_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData','logP_updates'])
top_calc_dir = os.path.sep.join([top_ds_dir,'Calc'])
all_stats_pkl = os.path.sep.join([top_calc_dir,'logP.All.Updates.Stats.pkl'])
all_raw_pkl = os.path.sep.join([top_calc_dir,'logP.All.Updates.Raw.pkl'])
###################################
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import ADMethod2Param,nrTrees_regression,native_uncertainty_alg_variant,ml_alg_regression,non_conformity_scaling,icp_calib_fraction,number_of_acp_splits,number_of_scp_splits,sig_level_of_interest,all_sig_levels_considered_to_compute_ECE
from consistent_parameters_for_all_modelling_runs import all_global_random_seed_opts
ml_alg = ml_alg_regression
nrTrees = nrTrees_regression
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_reg_uncertainty import funcsToApplyRegUncertaintyToNewDatasets as RegUncert
from modelsADuncertaintyPkg.utils.ML_utils import getXandY,singleRandomSplit
from modelsADuncertaintyPkg.qsar_eval import all_key_reg_stats_and_plots as RegEval
from modelsADuncertaintyPkg.utils.ML_utils import getXandY,singleRandomSplit,makeYLabelsNumeric
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,convertDefaultDictDictIntoDataFrame,neverEndingDefaultDict,findDups
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds,getInsideOutsideADSubsets
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import getFpsDfAndDataDfSkippingFailedMols
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,makeIDsNumeric,prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD,getADSubsetTestIDsInOrder
#----------------------------------------------------------------------------
#Dataset globals:
id_col = 'CSN' 
smiles_col = 'SMILES'
ep_col = 'logP'
dataset_file = os.path.sep.join([top_ds_dir,'logP_Data_Ready_for_Time_Splits_Anonymized.csv'])
#-----------------------------------------------------------------------------
#Dataset requirements for each update:
min_no_cmpds = 50 #Meaning each test set would need at least 50 compounds c.f. Sheridan (2022)
#--------------------------------------
updates_in_order = ['T{}'.format(t) for t in range(1,5)]
#--------------------------------------

#=================================
selected_uncertainty_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Regression']['uncertainty_method']]
selected_ad_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Regression']['AD_method_name']]
#=================================

def loadTrainTestSplit(dataset_file,update_version,id_col,smiles_col,ep_col):
    train_plus_test_plus_rest_df = pd.read_csv(dataset_file)
    
    dict_of_subsets = {}
    
    for subset in ['Train','Test']:
        dict_of_subsets[subset] = train_plus_test_plus_rest_df[train_plus_test_plus_rest_df['{} {}'.format(subset,update_version)].isin([True])]
        
        
        dict_of_subsets[subset] = dict_of_subsets[subset][[id_col,smiles_col,ep_col]]
    
    
    return dict_of_subsets['Train'],dict_of_subsets['Test']

def testSetHasEnoughCompounds(min_no_cmpds,test_df):
    if not test_df.shape[0] >= min_no_cmpds:
        return False
    else:
        return True

def populateDictWithStats(dict_of_stats,AD_method_name,uncertainty_method,AD_subset,rmse,MAD,R2,Pearson,Pearson_Pval_one_tail,Spearman,Spearman_Pval_one_tail,validity, efficiency,ECE_new,ENCE,no_compounds,scc):
    
    dict_of_current_stats = RegEval.map_all_regression_stats_onto_default_names(rmse, MAD, R2, Pearson, Pearson_Pval_one_tail, Spearman, Spearman_Pval_one_tail, validity, efficiency, ECE_new, ENCE,no_compounds,scc)
    
    for stat_name in dict_of_current_stats.keys():
        dict_of_stats[AD_method_name][uncertainty_method][AD_subset][stat_name] = dict_of_current_stats[stat_name]

def writeStatsFiles(dict_of_stats,calc_dir,selected_ad_methods,selected_uncertainty_methods,endpoint_name='logP'):
    
    statsFilesDict = defaultdict(dict)
    
    for AD_method_name in selected_ad_methods:
        for method in selected_uncertainty_methods:
            dd = dict_of_stats[AD_method_name][method]
            
            df = convertDefaultDictDictIntoDataFrame(dd,col_name_for_first_key='AD Subset')
            
            stats_file = os.path.sep.join([calc_dir,'e={}_ad={}_m={}_Regression_Statistics.csv'.format(endpoint_name,AD_method_name,method)])
            
            statsFilesDict[AD_method_name][method] = stats_file
            
            df.to_csv(stats_file,index=False)
    
    return statsFilesDict

def updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,AD_method_name,uncertainty_method,AD_subset,subset_test_ids,subset_test_y,subset_test_predictions,subset_sig_level_2_prediction_intervals):
    
    dict_of_raw_results[AD_method_name][uncertainty_method][AD_subset]['subset_test_ids'] = subset_test_ids
    dict_of_raw_results[AD_method_name][uncertainty_method][AD_subset]['subset_test_y'] = subset_test_y
    dict_of_raw_results[AD_method_name][uncertainty_method][AD_subset]['subset_test_predictions'] = subset_test_predictions
    dict_of_raw_results[AD_method_name][uncertainty_method][AD_subset]['subset_sig_level_2_prediction_intervals'] = subset_sig_level_2_prediction_intervals
    

def report_stats_for_model_ready_dataset(update_version,selected_uncertainty_methods,selected_ad_methods,ADMethod2Param,non_conformity_scaling,ml_alg,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,global_random_seed,id_col,smiles_col,ep_col,dataset_file,min_no_cmpds,sig_level_of_interest,all_sig_levels_considered_to_compute_ECE,native_uncertainty_alg_variant):
    print(f'Running report_stats_for_model_ready_dataset(...) for update_version={update_version},random seed={global_random_seed}')
    
    train_df,test_df = loadTrainTestSplit(dataset_file,update_version,id_col,smiles_col,ep_col)
    
    #==================================
    #Should not be needed for SYN datasets:
    train_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=train_df,smiles_col=smiles_col,activity_col=ep_col,unique_mol_id_col=id_col,type_of_activities="regression")
    
    test_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=test_df,smiles_col=smiles_col,activity_col=ep_col,unique_mol_id_col=id_col,type_of_activities="regression")
    #===================================
    
    checkTrainTestHaveUniqueDistinctIDs(train_df,test_df,id_col)
    
    if not testSetHasEnoughCompounds(min_no_cmpds,test_df):
        print('Skipping the rest of report_stats_for_model_ready_dataset(...) for update_version={}!'.format(update_version))
        return 0
    
    train_df = makeIDsNumeric(df=train_df,id_col=id_col)
    
    test_df = makeIDsNumeric(df=test_df,id_col=id_col)
    
    fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,ep_col,None)
    
    if not testSetHasEnoughCompounds(min_no_cmpds,X_test_and_ids_df):
        print('[After fingerprint calculation] Skipping the rest of report_stats_for_model_ready_dataset(...) for update_version={}!'.format(update_version))
        return 0
    
    #calc_dir = os.path.sep.join([top_calc_dir,f'{update_version}_s={global_random_seed}'])
    
    #createOrReplaceDir(dir_=calc_dir)
    
    train_inc_calib_x,train_inc_calib_y,test_x = prepareInputsForModellingAndUncertainty(X_train_and_ids_df,train_y,X_test_and_ids_df,id_col)
    
    print('='*50)
    print(f'update_version={update_version}')
    print(f'random seed={global_random_seed}')
    
    print(f'no. in training set={len(train_inc_calib_y)}')
    

    print(f'no. in test set={len(test_y)}')
    print('='*50)
    

def main():
    print('THE START')
    
    for update_version in updates_in_order:

        for global_random_seed in all_global_random_seed_opts:
        
            report_stats_for_model_ready_dataset(update_version,selected_uncertainty_methods,selected_ad_methods,ADMethod2Param,non_conformity_scaling,ml_alg,number_of_scp_splits,number_of_acp_splits,icp_calib_fraction,nrTrees,global_random_seed,id_col,smiles_col,ep_col,dataset_file,min_no_cmpds,sig_level_of_interest,all_sig_levels_considered_to_compute_ECE,native_uncertainty_alg_variant)
        
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())


