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
#Copyright (c) 2022-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Contact zied.hosni [at] syngenta.com
#############################################################
import os,   sys,   time,   pickle, shutil, itertools, statistics, re
from collections import defaultdict
import pandas as pd
import numpy as np
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_dir_of_public_data_scripts = os.path.dirname(dir_of_this_script)
##################################
sys.path.append(os.path.sep.join([top_dir_of_public_data_scripts,'general_purpose']))
from common_globals import consistent_id_col,   consistent_act_col,   consistent_smiles_col,   consistent_inchi_col
from common_globals import top_class_or_reg_ds_dirs,   regression_dataset_names,   classification_dataset_names,   ds_matched_to_ep_list,   get_endpoint_type
from common_globals import all_Wang_endpoints,   wang_raw_smiles_col,   wang_raw_act_col,   wang_ids_col,   wang_test_set_names,   no_wang_outer_folds
from common_globals import all_Tox21_endpoints,   all_ChEMBL_endpoints,   chembl_ids_col,   chembl_smiles_col,   chembl_act_class_col,   chembl_subsets_of_interest,   tox21_subsets_of_interest
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,   fp_col,   stand_mol_inchi_col,   stand_mol_smiles_col,   subset_name_col
from common_globals import get_rand_subset_train_test_suffix,   rand_split_seed,   no_train_test_splits
from common_globals import ds_matched_to_exemplar_ep_list
from common_globals import load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations
from common_globals import all_global_random_seed_opts
from common_globals import class_1,   class_0
from common_globals import rnd_test_suffix,  wang_test_set_names
from common_globals import class_ds_name_to_test_subset_names
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
##################################
from define_plot_settings import *
##################################
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,   convertDefaultDictDictIntoDataFrame,   neverEndingDefaultDict,   findDups
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import flatten
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,   makeIDsNumeric,   prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,   create_pkl_file, flatten
from modelsADuncertaintyPkg.utils.graph_utils import create_plot_comparing_two_distributions
#----------------------------------------------------------------------------
##################################
wang_model_ready_dataset_dir = os.path.sep.join([top_class_or_reg_ds_dirs['Wang_ChEMBL'],'Model_Ready'])

out_dir = os.path.sep.join([top_class_or_reg_ds_dirs['Wang_ChEMBL'],   'Activities'])
        
assert not out_dir == wang_model_ready_dataset_dir


def get_endpoint_specific_train_vs_test_activity_distributions(endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions,wang_test_set_names,wang_model_ready_dataset_dir, no_wang_outer_folds,endpoint,merged_split_label=''):
    dir_with_files_to_parse = wang_model_ready_dataset_dir

    for test_split_type in wang_test_set_names:
        
        all_train_y = []
        all_test_y = []

        for fold_no in range(no_wang_outer_folds):
            
            train_set_label = f'{endpoint}_{test_split_type}_f={fold_no}_train_FPs'
            test_set_label = f'{endpoint}_{test_split_type}_f={fold_no}_test_FPs'

            fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids = load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,  train_set_label,  test_set_label)

            train_y = train_y.tolist()
            test_y = test_y.tolist()
            
            endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][f'f={fold_no}']['train'] = train_y

            endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][f'f={fold_no}']['test'] = test_y

            all_train_y += train_y
            all_test_y += test_y
        
        #------------------------------
        #Perhaps this is merged distribution plot is redundant for cross-validation splits!
        endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][merged_split_label]['train'] = all_train_y

        endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][merged_split_label]['test'] = all_test_y
        #------------------------------

def get_per_endpoint_train_vs_test_activity_distributions(wang_model_ready_dataset_dir, no_wang_outer_folds,wang_test_set_names,ds_matched_to_ep_list):
    endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions = neverEndingDefaultDict()

    for endpoint in ds_matched_to_ep_list['Wang_ChEMBL']:
        get_endpoint_specific_train_vs_test_activity_distributions(endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions,wang_test_set_names,wang_model_ready_dataset_dir, no_wang_outer_folds,endpoint)


    return endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions





def plot_a_single_train_vs_test_activity_distributions(train_y,test_y,out_dir,endpoint,test_split_type,split_label,num_bins = 50,density_=True):
    if density_:
        ####################################
        #Probability density can exceed 1 if the bin width is less than 1, as the area under the histogram would still add up to 1.
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
        #https://stackoverflow.com/questions/55555466/matplotlib-hist-function-argument-density-not-working
        #####################################
        y_label = 'Probability density'
    else:
        raise Exception('Plotting histograms of raw counts would not make sense')

    plot_file_prefix = os.path.sep.join([out_dir,f'e={endpoint}_{test_split_type}_split={split_label}'])

    create_plot_comparing_two_distributions(plot_file_prefix,title=f'Activities: {plot_file_prefix}',prefix_label_1='Train',prefix_label_2='Test',dist_list_1=train_y,dist_list_2=test_y,x_label='Activities',y_label=y_label,density=density_,colour_1='red',colour_2='blue',num_bins=num_bins,add_median_line=True,add_mean_and_stdev=True,alpha=0.3)


def plot_train_vs_test_activity_distributions(endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions,out_dir):

    for endpoint in endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions.keys():
        for test_split_type in endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint].keys():
            for split_label in endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type].keys():
                train_y = endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][split_label]['train']

                test_y = endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions[endpoint][test_split_type][split_label]['test']

                plot_a_single_train_vs_test_activity_distributions(train_y,test_y,out_dir,endpoint,test_split_type,split_label)


def main():
    print('THE START')

    createOrReplaceDir(dir_=out_dir)

    endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions = get_per_endpoint_train_vs_test_activity_distributions(wang_model_ready_dataset_dir, no_wang_outer_folds,wang_test_set_names,ds_matched_to_ep_list)

    plot_train_vs_test_activity_distributions(endpoint_matched_to_split_type_to_fold_to_train_vs_test_activity_distributions,out_dir)


    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
