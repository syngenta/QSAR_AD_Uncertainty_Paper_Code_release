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
import os,   sys,   time,   pickle, shutil, itertools, statistics, re, glob
from collections import defaultdict
import pandas as pd
import numpy as np
##################################
from common_globals import consistent_id_col,   consistent_act_col,   consistent_smiles_col,   consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,   pkg_dir,   top_class_or_reg_ds_dirs,   regression_dataset_names,   classification_dataset_names,   ds_matched_to_ep_list,   get_endpoint_type
from common_globals import all_Wang_endpoints,   wang_raw_smiles_col,   wang_raw_act_col,   wang_ids_col,   wang_test_set_names,   no_wang_outer_folds
from common_globals import all_Tox21_endpoints,   all_ChEMBL_endpoints,   chembl_ids_col,   chembl_smiles_col,   chembl_act_class_col,   chembl_subsets_of_interest,   tox21_subsets_of_interest
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,   fp_col,   stand_mol_inchi_col,   stand_mol_smiles_col,   subset_name_col
from common_globals import get_rand_subset_train_test_suffix,   rand_split_seed,   no_train_test_splits
from common_globals import ds_matched_to_exemplar_ep_list
from common_globals import all_global_random_seed_opts
from common_globals import class_1,   class_0
from common_globals import rnd_test_suffix,  wang_test_set_names
from common_globals import class_ds_name_to_test_subset_names
from common_globals import overall_top_public_data_dir
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
from common_globals import rnd_train_suffix, rnd_test_suffix
from common_globals import check_model_ready_train_or_test_subset_counts_are_internally_consistent
##################################
#-----------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file
#-------------------------------------------

def remove_random_split_suffixes(train_or_test_set_label,rnd_train_suffix, rnd_test_suffix):
    return train_or_test_set_label.replace(f'_{rnd_train_suffix}','').replace(f'_{rnd_test_suffix}','')

def remove_fp_suffixes(train_or_test_set_label):
    return train_or_test_set_label.replace('_FPs','')

def remove_Wang_fold_specific_train_or_test_suffixes(train_or_test_set_label):
    return re.sub('(_f\=[0-9]+)','',train_or_test_set_label).replace('_train','').replace('_test','')

def check_remove_Wang_fold_specific_train_or_test_suffixes():
    assert 'A2a_ivit' == remove_Wang_fold_specific_train_or_test_suffixes('A2a_ivit_f=3_test'),remove_Wang_fold_specific_train_or_test_suffixes('A2a_ivit_f=3_test')

check_remove_Wang_fold_specific_train_or_test_suffixes()

def get_orig_subset_labels(ds_matched_to_ep_to_all_train_test_pairs,rnd_train_suffix, rnd_test_suffix):
    
    ds_matched_to_ep_to_orig_subset_labels = neverEndingDefaultDict()
    
    for dataset_name in ds_matched_to_ep_to_all_train_test_pairs.keys():
        for ep in ds_matched_to_ep_to_all_train_test_pairs[dataset_name].keys():
            all_labels = []
            
            for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][ep]:
                for label in train_test_label_pair:
                    all_labels.append(label)
            
            all_labels = [remove_random_split_suffixes(l,rnd_train_suffix, rnd_test_suffix) for l in all_labels]
            
            all_labels = [remove_fp_suffixes(l) for l in all_labels]

            if 'Wang_ChEMBL' == dataset_name:
                all_labels = [remove_Wang_fold_specific_train_or_test_suffixes(l) for l in all_labels]
            
            ds_matched_to_ep_to_orig_subset_labels[dataset_name][ep] = list(set(all_labels))
    
    return ds_matched_to_ep_to_orig_subset_labels

def check_no_matching_files(matching_files, dir_,subset_label,model_ready=False):
    if 'Wang_ChEMBL' in dir_:
        assert 10 == len(matching_files),f'matching_files={matching_files}'
    else:
        if 'Tox21Train' in subset_label:
            if model_ready:
                assert 2 == len(matching_files),f'matching_files={matching_files}'
            else:
                assert 1 == len(matching_files),f'matching_files={matching_files}'
        elif '_train' in subset_label:
            if model_ready:
                assert 2 == len(matching_files),f'matching_files={matching_files}'
            else:
                assert 1 == len(matching_files),f'matching_files={matching_files}'
        elif 'holdout' in subset_label or 'update1' in subset_label or 'Tox21Score' in subset_label or 'Tox21Test' in subset_label:
            assert 1 == len(matching_files),f'matching_files={matching_files}'
        else:
            raise Exception(f'Unexpected subset_label={subset_label}')

def identify_ds_subset_files(dir_,subset_label,model_ready=False):
    matching_files_without_specific_constraints = glob.glob(f'{os.path.sep.join([dir_,subset_label])}*') 

    matching_files = matching_files_without_specific_constraints

    check_no_matching_files(matching_files, dir_,subset_label,model_ready=model_ready)
    
    return matching_files

def add_classification_specific_stats(y_vals_list,ds,ep,subset_label,dict_to_update):
    
    dict_to_update[ds][ep][subset_label]['no. actives'] = len([v for v in y_vals_list if class_1 == v])
    
    dict_to_update[ds][ep][subset_label]['no. inactives'] = len([v for v in y_vals_list if class_0 == v])

    assert (dict_to_update[ds][ep][subset_label]['no. actives'] + dict_to_update[ds][ep][subset_label]['no. inactives'])==len(y_vals_list),f'ds={ds},ep={ep},subset_label={subset_label},y_vals_list={y_vals_list}'

def add_classification_specific_stats_for_pre_parsed_data(df, ds, ep, subset_label, all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col):
    y_vals_list = df[consistent_act_col].tolist()

    add_classification_specific_stats(y_vals_list=y_vals_list,ds=ds,ep=ep,subset_label=subset_label,dict_to_update=all_datasets_before_parsing_endpoint_to_orig_subset_to_stats)

def add_regression_specific_stats(y_vals_list,ds,ep,subset_label,dict_to_update):
    
    dict_to_update[ds][ep][subset_label]['minimum activity'] = min(y_vals_list)
    
    dict_to_update[ds][ep][subset_label]['maximum activity'] = max(y_vals_list)

    dict_to_update[ds][ep][subset_label]['median activity'] = np.median(y_vals_list)

    dict_to_update[ds][ep][subset_label]['mean activity'] = np.mean(y_vals_list)

    dict_to_update[ds][ep][subset_label]['s.d. activity'] = np.std(y_vals_list,ddof=0)

def add_regression_specific_stats_for_pre_parsed_data(df, ds, ep, subset_label, all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col):
    y_vals_list = df[consistent_act_col].tolist()

    add_regression_specific_stats(y_vals_list=y_vals_list,ds=ds,ep=ep,subset_label=subset_label,dict_to_update=all_datasets_before_parsing_endpoint_to_orig_subset_to_stats)

def get_a_single_data_frame_of_unique_ids(all_dfs_loaded_from_matching_files,consistent_id_col):
    merged_df = pd.concat(all_dfs_loaded_from_matching_files)

    merged_df.drop_duplicates(subset=consistent_id_col, inplace=True, ignore_index=True)

    return merged_df


def get_before_parsing_dataset_stats(dir_with_unfiltered_files_to_parse,ds_matched_to_ep_to_orig_subset_labels, dataset_name,endpoint_type,all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col, consistent_id_col):
    
    ds = dataset_name
    
    for ep in ds_matched_to_ep_to_orig_subset_labels[ds].keys():
        for subset_label in ds_matched_to_ep_to_orig_subset_labels[ds][ep]:
            
            matching_files = identify_ds_subset_files(dir_=dir_with_unfiltered_files_to_parse, subset_label=subset_label,model_ready=False)
            
            df = get_a_single_data_frame_of_unique_ids(all_dfs_loaded_from_matching_files=[pd.read_csv(f) for f in matching_files],consistent_id_col=consistent_id_col)
            
            all_datasets_before_parsing_endpoint_to_orig_subset_to_stats[ds][ep][subset_label]['no.compounds'] = df.shape[0]
            
            if 'classification' == endpoint_type:
                add_classification_specific_stats_for_pre_parsed_data(df, ds, ep, subset_label, all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col)
            elif 'regression' == endpoint_type:
                add_regression_specific_stats_for_pre_parsed_data(df, ds, ep, subset_label, all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col)
            else:
                raise Exception(f'ds={ds},ep={ep}: endpoint_type={endpoint_type}')
    
    return all_datasets_before_parsing_endpoint_to_orig_subset_to_stats

def extract_info_from_pkled_dict(pkled_dict,subset_type='train'):
    fps = pkled_dict[f'fps_{subset_type}']
    
    X_and_ids_df = pkled_dict[f'X_{subset_type}_and_ids_df']
    
    y = pkled_dict[f'{subset_type}_y']
    
    ids = pkled_dict[f'{subset_type}_ids']
    
    return fps,ids,X_and_ids_df,y

def load_from_model_ready_file(matched_file):
    matched_file_dict = load_from_pkl_file(matched_file)
    
    try:
        fps,ids,X_and_ids_df,y = extract_info_from_pkled_dict(pkled_dict=matched_file_dict,subset_type='train')
    except KeyError:
        fps,ids,X_and_ids_df,y = extract_info_from_pkled_dict(pkled_dict=matched_file_dict,subset_type='test')
    
    return fps,ids,X_and_ids_df,y

def add_classification_specific_stats_for_model_ready_data(complete_list_of_y_vals,ds,ep,subset_label, all_datasets_model_ready_endpoint_to_orig_subset_to_stats):

    add_classification_specific_stats(y_vals_list=complete_list_of_y_vals,ds=ds,ep=ep,subset_label=subset_label,dict_to_update=all_datasets_model_ready_endpoint_to_orig_subset_to_stats)

def add_regression_specific_stats_for_model_ready_data(complete_list_of_y_vals,ds,ep,subset_label, all_datasets_model_ready_endpoint_to_orig_subset_to_stats):

    add_regression_specific_stats(y_vals_list=complete_list_of_y_vals,ds=ds,ep=ep,subset_label=subset_label,dict_to_update=all_datasets_model_ready_endpoint_to_orig_subset_to_stats)

def get_model_ready_dataset_stats(dir_with_model_ready_files_to_parse,ds_matched_to_ep_to_orig_subset_labels, dataset_name,endpoint_type,all_datasets_model_ready_endpoint_to_orig_subset_to_stats, consistent_act_col):
    
    ds = dataset_name
    
    for ep in ds_matched_to_ep_to_orig_subset_labels[ds].keys():
        for subset_label in ds_matched_to_ep_to_orig_subset_labels[ds][ep]:
            
            matching_files = identify_ds_subset_files(dir_=dir_with_model_ready_files_to_parse, subset_label=subset_label,model_ready=True)
            
            id2Yval = {}
            
            for matched_file in matching_files:
                fps,ids,X_and_ids_df,y = load_from_model_ready_file(matched_file)
                
                check_model_ready_train_or_test_subset_counts_are_internally_consistent(fps,ids,X_and_ids_df,y)
                
                for index in range(len(ids)):
                    id2Yval[ids[index]]=y.tolist()[index]
            
            complete_list_of_y_vals = list(id2Yval.values())

            all_datasets_model_ready_endpoint_to_orig_subset_to_stats[ds][ep][subset_label]['no.compounds'] = len(complete_list_of_y_vals)
            
            if 'classification' == endpoint_type:
                add_classification_specific_stats_for_model_ready_data(complete_list_of_y_vals,ds,ep,subset_label, all_datasets_model_ready_endpoint_to_orig_subset_to_stats)
            elif 'regression' == endpoint_type:
                add_regression_specific_stats_for_model_ready_data(complete_list_of_y_vals,ds,ep,subset_label, all_datasets_model_ready_endpoint_to_orig_subset_to_stats)
            else:
                raise Exception(f'ds={ds},ep={ep}: endpoint_type={endpoint_type}')
    
    return all_datasets_model_ready_endpoint_to_orig_subset_to_stats

def write_dataset_statistics_file(out_file,dataset_name,endpoint_type,all_datasets_before_parsing_endpoint_to_orig_subset_to_stats,all_datasets_model_ready_endpoint_to_orig_subset_to_stats):

    relevant_dataset_dict_before_parsing = all_datasets_before_parsing_endpoint_to_orig_subset_to_stats[dataset_name]

    relevant_dataset_dict_after_parsing = all_datasets_model_ready_endpoint_to_orig_subset_to_stats[dataset_name]

    ep_col = 'Endpoint (Target) Name'
    sub_col = 'Subset Name'
    status_col = 'Parsing Status'
    orig_status = 'Unfiltered'
    model_ready_status = 'Model ready'

    output_cols_in_addition_to_statistics = [ep_col,sub_col,status_col]

    table_precursor_dict = defaultdict(list)

    for ep in relevant_dataset_dict_before_parsing.keys():
        for subset_label in relevant_dataset_dict_before_parsing[ep].keys():
            for parsing_status in [orig_status,model_ready_status]:
                if orig_status == parsing_status:
                    stats_dict = relevant_dataset_dict_before_parsing[ep][subset_label]
                elif model_ready_status == parsing_status:
                    stats_dict = relevant_dataset_dict_after_parsing[ep][subset_label]
                else:
                    raise Exception(f'Unrecognised status={parsing_status}')
                
                table_precursor_dict[ep_col].append(ep)
                table_precursor_dict[sub_col].append(subset_label)
                table_precursor_dict[status_col].append(parsing_status)

                for stat_name in stats_dict.keys():
                    table_precursor_dict[stat_name].append(stats_dict[stat_name])

    stats_table_df = pd.DataFrame(table_precursor_dict)

    stats_table_df.to_excel(out_file,index=False,engine='openpyxl')
                


def main(debug=False):
    print('THE START')
    
    ds_matched_to_ep_to_orig_subset_labels = get_orig_subset_labels(ds_matched_to_ep_to_all_train_test_pairs,rnd_train_suffix, rnd_test_suffix)
    
    if debug:
        for ds in ds_matched_to_ep_to_orig_subset_labels.keys():
            for ep in ds_matched_to_ep_to_orig_subset_labels[ds].keys():
                for subset_label in ds_matched_to_ep_to_orig_subset_labels[ds][ep]:
                    print(f'Dataset={ds},ep={ep},subset_label={subset_label}')
    
    out_dir = os.path.sep.join([overall_top_public_data_dir,   'DataStats'])
    
    
    createOrReplaceDir(dir_=out_dir)
    
    all_datasets_before_parsing_endpoint_to_orig_subset_to_stats = neverEndingDefaultDict()
    
    all_datasets_model_ready_endpoint_to_orig_subset_to_stats = neverEndingDefaultDict()
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        out_file = os.path.sep.join([out_dir,f'{dataset_name}_SummaryStatistics.xlsx'])

        endpoint_type = get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)
        
        dir_with_unfiltered_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'NoFilt'])
        
        dir_with_model_ready_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        #-------------------------------------
        assert not dir_with_unfiltered_files_to_parse == dir_with_model_ready_files_to_parse, dir_with_model_ready_files_to_parse
        #-------------------------------------
        
        ##########################
        #orig_subset means prior to our random split of Morger_ChEMBL and Morger_Tox21 training splits:
        all_datasets_before_parsing_endpoint_to_orig_subset_to_stats = get_before_parsing_dataset_stats(dir_with_unfiltered_files_to_parse,ds_matched_to_ep_to_orig_subset_labels, dataset_name,endpoint_type,all_datasets_before_parsing_endpoint_to_orig_subset_to_stats, consistent_act_col, consistent_id_col) 
        
        all_datasets_model_ready_endpoint_to_orig_subset_to_stats = get_model_ready_dataset_stats(dir_with_model_ready_files_to_parse,ds_matched_to_ep_to_orig_subset_labels, dataset_name,endpoint_type,all_datasets_model_ready_endpoint_to_orig_subset_to_stats, consistent_act_col)
        ##########################

        write_dataset_statistics_file(out_file,dataset_name,endpoint_type,all_datasets_before_parsing_endpoint_to_orig_subset_to_stats,all_datasets_model_ready_endpoint_to_orig_subset_to_stats)
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
