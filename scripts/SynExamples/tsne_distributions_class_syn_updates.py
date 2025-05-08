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
top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData','DT50_Updates'])
top_calc_dir = os.path.sep.join([top_ds_dir,'TSNE'])
###################################
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import ADMethod2Param, ml_alg_classification,lit_precedent_delta_for_calib_plot,larger_delta_for_calib_plot
from consistent_parameters_for_all_modelling_runs import all_global_random_seed_opts
ml_alg = ml_alg_classification
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_class_uncertainty import define_native_ML_baseline as nml
from modelsADuncertaintyPkg.qsar_class_uncertainty import IVAP_CVAP_workflow_functions as vap
from modelsADuncertaintyPkg.qsar_eval import all_key_class_stats_and_plots as ClassEval
from modelsADuncertaintyPkg.qsar_class_uncertainty.parse_native_or_venn_abers_output import getProbsForClass1
from modelsADuncertaintyPkg.utils.load_example_datasets_for_code_checks import generateExampleDatasetForChecking_BCWD
from modelsADuncertaintyPkg.utils.ML_utils import getXandY,singleRandomSplit,makeYLabelsNumeric
from modelsADuncertaintyPkg.utils.ML_utils import predictedBinaryClassFromProbClass1
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import findDups,convertDefaultDictDictIntoDataFrame,neverEndingDefaultDict,findDups
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import findInsideOutsideADTestIds,getInsideOutsideADSubsets
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import getFpsDfAndDataDfSkippingFailedMols
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,makeIDsNumeric,prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD,getADSubsetTestIDsInOrder
#----------------------------------------------------------------------------
from modelsADuncertaintyPkg.CheminformaticsUtils.visualize_chemical_space import get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels
from modelsADuncertaintyPkg.CheminformaticsUtils.visualize_chemical_space import prepare_X_and_ids_df_for_TSNE
#----------------------------------------------------------------------------
from class_syn_updates import id_col,smiles_col,class_label_col,original_class_1,original_class_0,class_1,class_0,min_no_cmpds_per_class,updates_in_order,update2DataFiles

from class_syn_updates import updateClassLabels,prepareClassData_OneFile,prepareClassData,testSetHasEnoughCompoundsPerClass


def compute_tsne_plot_for_one_timesplit(update_version,top_calc_dir,min_no_cmpds_per_class=min_no_cmpds_per_class,id_col=id_col,smiles_col=smiles_col,class_label_col=class_label_col,class_1=class_1,original_class_1=original_class_1,original_class_0=original_class_0,subset_col='Subset'):
    
    print(f'Running compute_tsne_plot_for_one_timesplit(...) for update_version={update_version}')
    
    train_file = update2DataFiles[update_version]['train_file']
    
    test_file = update2DataFiles[update_version]['test_file']
    
    train_df,test_df = prepareClassData(train_file,test_file,id_col,smiles_col,class_label_col,class_1,original_class_1,original_class_0)
    
    #==================================
    #Should not be needed for SYN datasets:
    train_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=train_df,smiles_col=smiles_col,activity_col=class_label_col,unique_mol_id_col=id_col,type_of_activities="classification")
    
    test_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=test_df,smiles_col=smiles_col,activity_col=class_label_col,unique_mol_id_col=id_col,type_of_activities="classification")
    #===================================
    
    checkTrainTestHaveUniqueDistinctIDs(train_df,test_df,id_col)
    
    if not testSetHasEnoughCompoundsPerClass(min_no_cmpds_per_class,test_df,class_label_col,class_1,original_class_0,test_y=None):
        print('Skipping the rest of compute_tsne_plot_for_one_timesplit(...) for update_version={}!'.format(update_version))
        return 0
    
    train_df = makeIDsNumeric(df=train_df,id_col=id_col)
    
    test_df = makeIDsNumeric(df=test_df,id_col=id_col)
    
    fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col,smiles_col,class_label_col,class_1)
    
    if not testSetHasEnoughCompoundsPerClass(min_no_cmpds_per_class,None,class_label_col,class_1,original_class_0,test_y):
        print('[After fingerprint calculation] Skipping the rest of compute_tsne_plot_for_one_timesplit(...) for update_version={}!'.format(update_version))
        return 0
    
    test_set_label = 'Test'
    train_set_label = 'Train'

    test_df_ready_for_tsne = prepare_X_and_ids_df_for_TSNE(X_and_ids_df=X_test_and_ids_df,subset_col=subset_col,subset_name=test_set_label,id_col=id_col)

    train_df_ready_for_tsne = prepare_X_and_ids_df_for_TSNE(X_and_ids_df=X_train_and_ids_df,subset_col=subset_col,subset_name=train_set_label,id_col=id_col)

    df_ready_for_tsne = pd.concat([test_df_ready_for_tsne,train_df_ready_for_tsne])
    df_ready_for_tsne.reset_index(drop=True,inplace=True)

    plot_name_prefix = os.path.sep.join([top_calc_dir,update_version])

    title_prefix = update_version

    tsne_start = time.time()

    get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels(plot_name_prefix,title_prefix,dataset_df=df_ready_for_tsne,subset_col=subset_col)

    tsne_end = time.time()

    print('='*50)
    print('Computed TSNE for:')
    print(update_version)
    print(f'Time taken = {(tsne_end-tsne_start)/60} minutes')
    print('='*50)

def main():
    print('THE START')

    createOrReplaceDir(dir_=top_calc_dir)
    
    for update_version in updates_in_order:

        compute_tsne_plot_for_one_timesplit(update_version,top_calc_dir)
        
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
