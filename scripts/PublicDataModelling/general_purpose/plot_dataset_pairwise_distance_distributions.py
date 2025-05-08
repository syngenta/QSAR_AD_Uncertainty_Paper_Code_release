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
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#Contact zied.hosni [at] syngenta.com
#############################################################
#########################
#def FreqTanimotoSplitting(...) was copied from Zied's extraFunctions.py and renamed plot_train_vs_train_compared_to_train_vs_test_Tanimoto_distances(...)
#Subsequently,  the function was adapted,  including splitting parts of the code into smaller functions,  where appropriate.
########################
import os,   sys,   time,   pickle, shutil, itertools, statistics, re
from collections import defaultdict
import pandas as pd
import numpy as np
from textwrap import wrap
##################################
from common_globals import consistent_id_col,   consistent_act_col,   consistent_smiles_col,   consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,   pkg_dir,   top_class_or_reg_ds_dirs,   regression_dataset_names,   classification_dataset_names,   ds_matched_to_ep_list,   get_endpoint_type
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
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.ML_utils import getTestYFromTestIds as getSubsetYFromSubsetIds
from modelsADuncertaintyPkg.utils.basic_utils import check_findDups,   convertDefaultDictDictIntoDataFrame,   neverEndingDefaultDict,   findDups
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import flatten
from modelsADuncertaintyPkg.utils.ML_utils import checkTrainTestHaveUniqueDistinctIDs,   makeIDsNumeric,   prepareInputsForModellingAndUncertainty
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,   create_pkl_file, flatten
from modelsADuncertaintyPkg.CheminformaticsUtils.dataset_distributions import compute_ds1_vs_ds2_pairwise_distances, compute_pairwise_distances_for_dataset
#----------------------------------------------------------------------------
##################################
consistent_delim = 'NotAlreadyPresent' #Prevent bug caused by splitting on delim='.'!


def  get_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics(dataset_name,  dir_with_files_to_parse,   out_dir,   ds_matched_to_relevant_ep_list,   ds_matched_to_ep_to_all_train_test_pairs,delim=consistent_delim):
    
    res_pkl_file = os.path.sep.join([out_dir, 'Distances.pkl'])
    
    dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics = neverEndingDefaultDict()
    
    for endpoint in ds_matched_to_relevant_ep_list[dataset_name]:
        
        seen_train_labels = []
        
        for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
            
            print('='*20)
            print('Computing Tanimoto distances for:')
            print(f'dataset_name={dataset_name}, endpoint={endpoint}')
            print(f'train_test_label_pair={train_test_label_pair}')
            print('='*20)
            
            start = time.time()
            
            train_set_label = train_test_label_pair[0]
            test_set_label = train_test_label_pair[1]
            
            fps_train,   fps_test,   test_ids,   X_train_and_ids_df,   X_test_and_ids_df,   train_y,   test_y,   train_ids =load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,   train_set_label,   test_set_label)
            
            
            train_vs_test_pairwise_distances, dict_of_id_pairs_to_train_vs_test_pairwise_distances = compute_ds1_vs_ds2_pairwise_distances(fps_df_for_ds1=fps_train, data_ids_for_ds1=train_ids, fps_df_for_ds2=fps_test, data_ids_for_ds2=test_ids)
            
            if not train_set_label in seen_train_labels:
                train_vs_train_pairwise_distances, dict_of_train_id_pairs_to_pairwise_distances = compute_pairwise_distances_for_dataset(fps_df_for_dataset=fps_train, data_ids_for_dataset=train_ids)
                
                seen_train_labels.append(train_set_label)
            else:
                pass
            
            dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][delim.join(train_test_label_pair)]['train_vs_test_pairwise_distances'] = train_vs_test_pairwise_distances
            
            dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][delim.join(train_test_label_pair)]['dict_of_id_pairs_to_train_vs_test_pairwise_distances'] = dict_of_id_pairs_to_train_vs_test_pairwise_distances
            
            dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][delim.join(train_test_label_pair)]['train_vs_train_pairwise_distances'] = train_vs_train_pairwise_distances
            
            dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][delim.join(train_test_label_pair)]['dict_of_train_id_pairs_to_pairwise_distances'] = dict_of_train_id_pairs_to_pairwise_distances
            
            #Commented out to try and avoid MemoryError: #create_pkl_file(pkl_file=res_pkl_file,   obj=dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics) #Keep saving this to help with debugging in case of an intermediate crash
            
            end = time.time()
            
            print('='*20)
            print('COMPUTED Tanimoto distances for:')
            print(f'dataset_name={dataset_name}, endpoint={endpoint}')
            print(f'train_test_label_pair={train_test_label_pair}')
            print('='*20)
            
            print(f'Time taken = {(end-start)/60} minutes')
    
    return dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics

def get_fold(train_test_label_pair,dataset_name,merged_across_folds):
    if not 'Wang_ChEMBL' == dataset_name or merged_across_folds:
        fold_ = 'N/A'
    else:
        try:
            fold_ = int(train_test_label_pair.split('_f=')[1].split('_')[0])
        except Exception as err:
            raise Exception(f'Problem extracting fold from train_test_label_pair={train_test_label_pair}')
    
    return fold_

def plot_train_vs_train_compared_to_train_vs_test_Tanimoto_distances(train_vs_train_pairwise_distances, train_vs_test_pairwise_distances, raw_label_train, raw_label_test, endpoint, plot_file_prefix, fold=None, num_bins = 100, x_label="Tanimoto distance", density=True,legend_size=5):    
    label_train = 'Train' #raw_label_train.replace("_",  " " )
    if not 'N/A' == fold:
        label_test = f'Test (f={fold})' #raw_label_test.replace("_",  " " )
    else:
        label_test = 'Test'

    
    plt.hist(train_vs_test_pairwise_distances,  num_bins,  facecolor='blue',  alpha=0.3,  label=f'{label_train} vs. {label_test}',  density=density)
    
    plt.hist(train_vs_train_pairwise_distances,  num_bins,  facecolor='red',  alpha=0.3,  label=label_train,  density=density)
    
    title = 'Test set={},  fold={} \n {}'.format(raw_label_test,  fold,  endpoint)
    plt.title(title)
    
    median_train_vs_test_pairwise_distances = statistics.median(train_vs_test_pairwise_distances)
    median_train_vs_train_pairwise_distances = statistics.median(train_vs_train_pairwise_distances)
    
    plt.axvline(median_train_vs_test_pairwise_distances,  color='blue',  linestyle='dashed',  linewidth=1)
    plt.axvline(median_train_vs_train_pairwise_distances,  color='red',  linestyle='dashed',  linewidth=1)
    
    plt.legend(prop={'size': legend_size})
    plt.xlabel(x_label)
    
    if not density:
        plt.ylabel("Number of compound pairs")
        plot_file = f'{plot_file_prefix}_raw.png'
    else:
        ####################################
        #Probability density can exceed 1 if the bin width is less than 1, as the area under the histogram would still add up to 1.
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
        #https://stackoverflow.com/questions/55555466/matplotlib-hist-function-argument-density-not-working
        #####################################
        plt.ylabel("Probability density of compound pairs")
        plot_file = f'{plot_file_prefix}_fract.png'
        
    
    plt.tight_layout()
    plt.savefig(plot_file,  dpi=300,  bbox_inches="tight",  transparent=True)
    plt.close()
    plt.clf()

def plot_all_pairwise_distance_distributions(dataset_name, out_dir, dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics,delim=consistent_delim,merged_across_folds=False):
    for density_ in [True, False]:
        
        for endpoint in dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics.keys():
            for train_test_label_pair in dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint].keys():
                
                fold_ = get_fold(train_test_label_pair,dataset_name,merged_across_folds)
                
                plot_file_prefix = os.path.sep.join([out_dir,f'ep={endpoint}_split={train_test_label_pair}']) 
                
                train_vs_train_pairwise_distances = dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][train_test_label_pair]['train_vs_train_pairwise_distances']
                
                train_vs_test_pairwise_distances = dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][train_test_label_pair]['train_vs_test_pairwise_distances']
                
                raw_label_train, raw_label_test = train_test_label_pair.split(delim)
                
                plot_train_vs_train_compared_to_train_vs_test_Tanimoto_distances(train_vs_train_pairwise_distances, train_vs_test_pairwise_distances, raw_label_train, raw_label_test, endpoint, plot_file_prefix, fold=fold_, density=density_)

def merge_distributions_across_folds(dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics,delim=consistent_delim,no_folds=no_wang_outer_folds,all_test_type_names=wang_test_set_names):
    
    dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics = neverEndingDefaultDict()
    
    for endpoint in dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics.keys():
        
        
        
        for test_type_name in all_test_type_names:
            
            list_of_lists_train_vs_train_pairwise_distances = []
            
            list_of_lists_train_vs_test_pairwise_distances = []
            
            for train_test_label_pair in dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint].keys():
                if f'_{test_type_name}_' in train_test_label_pair: #c.f. def define_all_train_test_split_labels(...): in common_globals.py
                    
                    list_of_lists_train_vs_train_pairwise_distances.append(dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][train_test_label_pair]['train_vs_train_pairwise_distances'])
                    
                    list_of_lists_train_vs_test_pairwise_distances.append(dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics[endpoint][train_test_label_pair]['train_vs_test_pairwise_distances'])
            
            #---------------------------------
            assert no_folds == len(list_of_lists_train_vs_train_pairwise_distances),f'test_type_name={test_type_name},len(list_of_lists_train_vs_train_pairwise_distances)={len(list_of_lists_train_vs_train_pairwise_distances)}'
            assert no_folds == len(list_of_lists_train_vs_test_pairwise_distances),f'test_type_name={test_type_name},len(list_of_lists_train_vs_test_pairwise_distances)={len(list_of_lists_train_vs_test_pairwise_distances)}'
            #---------------------------------
            
            merged_folds_train_test_pair_label = delim.join(['Merged_Training_Sets',f'{test_type_name}_Test_Sets'])
            
            dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics[endpoint][merged_folds_train_test_pair_label]['train_vs_test_pairwise_distances'] = flatten(list_of_lists_train_vs_test_pairwise_distances)
            
            dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics[endpoint][merged_folds_train_test_pair_label]['train_vs_train_pairwise_distances'] = flatten(list_of_lists_train_vs_train_pairwise_distances)
            
            
    return dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics

def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Distributions'])
        
        assert not out_dir == dir_with_files_to_parse,   dir_with_files_to_parse
        
        createOrReplaceDir(dir_=out_dir)
        
        dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics = get_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics(dataset_name,  dir_with_files_to_parse,   out_dir,   ds_matched_to_relevant_ep_list=ds_matched_to_ep_list,   ds_matched_to_ep_to_all_train_test_pairs=ds_matched_to_ep_to_all_train_test_pairs)
        
        plot_all_pairwise_distance_distributions(dataset_name, out_dir, dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics,merged_across_folds=False)
        
        if 'Wang_ChEMBL' == dataset_name:
            dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics = merge_distributions_across_folds(dict_of_all_per_endpoint_per_train_test_split_tanimoto_distances_and_statistics)
            
            plot_all_pairwise_distance_distributions(dataset_name, out_dir, dict_of_all_per_endpoint_per_merged_train_test_split_tanimoto_distances_and_statistics,merged_across_folds=True)
            
            #plot_endpoint_value_distributions(dataset_name,  dir_with_files_to_parse,   out_dir,   ds_matched_to_relevant_ep_list,   ds_matched_to_ep_to_all_train_test_pairs)
        
        
        
        
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

