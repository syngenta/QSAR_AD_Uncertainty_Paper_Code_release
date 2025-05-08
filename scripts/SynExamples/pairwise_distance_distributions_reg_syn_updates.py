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
top_calc_dir = os.path.sep.join([top_ds_dir,'Dist'])
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
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.CheminformaticsUtils.dataset_distributions import compute_ds1_vs_ds2_pairwise_distances, compute_pairwise_distances_for_dataset
#----------------------------------------------------------------------------
#Also, import distance distributions plot function from script originally used to plot these for public data
#If we try to use relative imports, this causes problems with imports inside the script we are importing from
sys.path.append(os.path.sep.join([top_scripts_dir,'PublicDataModelling','general_purpose']))
from plot_dataset_pairwise_distance_distributions import plot_train_vs_train_compared_to_train_vs_test_Tanimoto_distances
#----------------------------------------------------------------------------

from reg_syn_updates import id_col,smiles_col,ep_col,dataset_file,min_no_cmpds,updates_in_order
from reg_syn_updates import loadTrainTestSplit,testSetHasEnoughCompounds



def distance_distributions_for_model_ready_dataset(update_version,id_col,smiles_col,ep_col,dataset_file,min_no_cmpds,top_calc_dir):
    print(f'Running distance_distributions_for_model_ready_dataset(...) for update_version={update_version}')
    
    train_df,test_df = loadTrainTestSplit(dataset_file,update_version,id_col,smiles_col,ep_col)
    
    #==================================
    #Should not be needed for SYN datasets:
    train_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=train_df,smiles_col=smiles_col,activity_col=ep_col,unique_mol_id_col=id_col,type_of_activities="regression")
    
    test_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=test_df,smiles_col=smiles_col,activity_col=ep_col,unique_mol_id_col=id_col,type_of_activities="regression")
    #===================================
    
    checkTrainTestHaveUniqueDistinctIDs(train_df,test_df,id_col)
    
    if not testSetHasEnoughCompounds(min_no_cmpds,test_df):
        print('Skipping the rest of distance_distributions_for_model_ready_dataset(...) for update_version={}!'.format(update_version))
        return 0
    
    train_df = makeIDsNumeric(df=train_df,id_col=id_col)
    
    test_df = makeIDsNumeric(df=test_df,id_col=id_col)
    
    fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,ep_col,None)
    
    if not testSetHasEnoughCompounds(min_no_cmpds,X_test_and_ids_df):
        print('[After fingerprint calculation] Skipping the rest of distance_distributions_for_model_ready_dataset(...) for update_version={}!'.format(update_version))
        return 0
    
    
    train_vs_test_pairwise_distances, dict_of_id_pairs_to_train_vs_test_pairwise_distances = compute_ds1_vs_ds2_pairwise_distances(fps_df_for_ds1=fps_train, data_ids_for_ds1=train_ids, fps_df_for_ds2=fps_test, data_ids_for_ds2=test_ids)
            
    train_vs_train_pairwise_distances, dict_of_train_id_pairs_to_pairwise_distances = compute_pairwise_distances_for_dataset(fps_df_for_dataset=fps_train, data_ids_for_dataset=train_ids)

    #=================
    #----------------------------
    #This is actually redundant:
    raw_label_train = None
    #----------------------------

    raw_label_test = update_version

    endpoint = ep_col

    plot_file_prefix = os.path.sep.join([top_calc_dir,f'{endpoint}_{update_version}_train_vs_test_distances'])



    plot_train_vs_train_compared_to_train_vs_test_Tanimoto_distances(train_vs_train_pairwise_distances, train_vs_test_pairwise_distances, raw_label_train, raw_label_test, endpoint, plot_file_prefix)      
    #=================

def main():
    print('THE START')

    createOrReplaceDir(dir_=top_calc_dir)
    
    for update_version in updates_in_order:

        distance_distributions_for_model_ready_dataset(update_version,id_col,smiles_col,ep_col,dataset_file,min_no_cmpds,top_calc_dir)
        
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())


