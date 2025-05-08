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
from common_globals import get_ds_matched_to_other_endpoints_of_interest
##################################
from define_plot_settings import *
##################################
#---------------------------------
sys.path.append(top_scripts_dir)
#---------------------------------
from consistent_parameters_for_all_modelling_runs import ADMethod2Param, ml_alg_classification,lit_precedent_delta_for_calib_plot,larger_delta_for_calib_plot
ml_alg = ml_alg_classification
from recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
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
selected_uncertainty_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Classification']['uncertainty_method']]
selected_ad_methods = [ep_type_matched_to_default_AD_uncertainty_methods['Classification']['AD_method_name']]
#=================================

from perform_classification_modelling_on_exemplar_endpoints import get_nested_dicts_of_modelling_results_for_all_exemplar_endpoints as get_nested_dicts_of_modelling_results_for_all_other_endpoints

def filter_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list(ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,endpoints_of_interest):

    new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = defaultdict(list)

    for ds in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list.keys():
        for ep in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds]:
            if ep in endpoints_of_interest:
                new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds].append(ep)

    return new_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list

def main():
    print('THE START')

    ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = get_ds_matched_to_other_endpoints_of_interest(ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list)

    #####################################
    parser = argparse.ArgumentParser(
        description='Specify endpoint to generate results for in case running all endpoints at once takes too long.')
    parser.add_argument('-e', dest="endpoint_of_interest", action='store', type=str,
                        help='Specify endpoint to generate results for', default='.'.join(list(itertools.chain(*[[e for e in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds]] for ds in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list.keys()]))))
    ##############################
    ##############################
    dict_of_opts_which_can_be_set_from_cmdline = vars(parser.parse_args())
    print('*'*50)
    print('Running calculations with the following options specified:')
    for var_name in dict_of_opts_which_can_be_set_from_cmdline.keys():
        print(var_name, '=',
              dict_of_opts_which_can_be_set_from_cmdline[var_name])
    print('*'*50)
    
    endpoint_of_interest = dict_of_opts_which_can_be_set_from_cmdline['endpoint_of_interest']
    #############################
    
    ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = filter_ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list(ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list,endpoints_of_interest=[endpoint_of_interest])
    
    for dataset_name in ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list:
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        top_calc_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Modelling.2'])
        
        assert not top_calc_dir == dir_with_files_to_parse,   dir_with_files_to_parse
        
        #createOrReplaceDir(dir_=top_calc_dir)

        calc_dir = os.path.sep.join([top_calc_dir, f"{endpoint_of_interest}_k={ADMethod2Param['UNC']}_t={ADMethod2Param['Tanimoto']}"])
        
        createOrReplaceDir(dir_=calc_dir)
        
        dict_of_raw_results = neverEndingDefaultDict()
        
        dict_of_stats = neverEndingDefaultDict()
        
        statsFilesDict = neverEndingDefaultDict()

        dict_of_raw_results, dict_of_stats, statsFilesDict = get_nested_dicts_of_modelling_results_for_all_other_endpoints(dataset_name, dict_of_stats, statsFilesDict, dict_of_raw_results, ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list, selected_uncertainty_methods, selected_ad_methods, all_global_random_seed_opts, consistent_id_col,   consistent_act_col, dir_with_files_to_parse, calc_dir, class_1, class_0)
    
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
