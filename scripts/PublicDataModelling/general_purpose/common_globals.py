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
#Copyright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys
import pandas as pd
from collections import defaultdict

consistent_id_col = "Mol_ID"
consistent_act_col = "Modelled_Endpoint"
consistent_smiles_col = "Mol_SMILES"
consistent_inchi_col = "Mol_InChI_ID"

stand_mol_col = 'StandardizedMol'
fp_col = 'Fingerprint'
stand_mol_inchi_col = "StandardizedMol_InChI_ID"
stand_mol_smiles_col='StandardizedMol_smiles'
subset_name_col = "Subset"

class_1 = 1
class_0 = 0



#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
top_dir_of_public_data_scripts = os.path.dirname(dir_of_this_file)
top_dir_of_all_scripts = os.path.dirname(top_dir_of_public_data_scripts)
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_file)))
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import doubleDefaultDictOfLists,findDups
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,create_pkl_file
from modelsADuncertaintyPkg.CheminformaticsUtils.chem_data_parsing_utils import check_fps_df
from modelsADuncertaintyPkg.utils.basic_utils import findDups,convertDefaultDictDictIntoDataFrame,neverEndingDefaultDict
#-----------------------------------------------
overall_top_public_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])
top_class_or_reg_ds_dirs = {}
top_class_or_reg_ds_dirs['Morger_Tox21'] = os.path.sep.join([overall_top_public_data_dir,'Tox21'])
top_class_or_reg_ds_dirs['Morger_ChEMBL'] = os.path.sep.join([overall_top_public_data_dir,'Morger_ChEMBL'])
top_class_or_reg_ds_dirs['Wang_ChEMBL'] = os.path.sep.join([overall_top_public_data_dir,'Wang_ChEMBL','data','dataset'])
regression_dataset_names = ['Wang_ChEMBL']
classification_dataset_names = ['Morger_ChEMBL','Morger_Tox21']
#----------------------------------------------
sys.path.append(top_dir_of_public_data_scripts)
from regression.common_globals_regression_scripts import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_set_names,no_wang_outer_folds,exemplar_Wang_endpoints
from classification.common_globals_classification_scripts import all_Tox21_endpoints,all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest,tox21_subsets_of_interest,rand_test_split_fract,exemplar_ChEMBL_endpoints,exemplar_Tox21_endpoints
from classification.common_globals_classification_scripts import ds_name_to_train_subset_name as class_ds_name_to_train_subset_name
from classification.common_globals_classification_scripts import ds_name_to_test_subset_names as class_ds_name_to_test_subset_names
#-------------------------------------------
print(f'top_dir_of_all_scripts={top_dir_of_all_scripts}')
sys.path.append(top_dir_of_all_scripts)
from consistent_parameters_for_all_modelling_runs import all_global_random_seed_opts
print(f'all_global_random_seed_opts = {all_global_random_seed_opts}')
#-------------------------------------------
ds_matched_to_ep_list = {}
ds_matched_to_ep_list['Morger_Tox21'] = all_Tox21_endpoints
ds_matched_to_ep_list['Morger_ChEMBL'] = all_ChEMBL_endpoints
ds_matched_to_ep_list['Wang_ChEMBL'] = all_Wang_endpoints
#-------------------------------------------
ds_matched_to_exemplar_ep_list={}
ds_matched_to_exemplar_ep_list['Morger_Tox21'] = exemplar_Tox21_endpoints
ds_matched_to_exemplar_ep_list['Morger_ChEMBL'] = exemplar_ChEMBL_endpoints
ds_matched_to_exemplar_ep_list['Wang_ChEMBL'] = exemplar_Wang_endpoints
#-------------------------------------------
def get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names):
    
    if dataset_name in regression_dataset_names:
        endpoint_type = 'regression'
    elif dataset_name in classification_dataset_names:
        endpoint_type = 'classification'
    else:
        raise Exception(f'Cannot figure out endpoint_type from this dataset_name={dataset_name}')
    
    return endpoint_type
#-----------------------------------------
rand_split_seed=all_global_random_seed_opts[0]
no_train_test_splits=1 #Size of Tox21 and ChEMBL train sets means a single random train:test split should be sufficient to get a robust result
#-----------------------------------------
act_col_to_use_for_filtering_based_upon_stand_mol_inchis = 'TmpAct'
#------------------------------------------
def get_rand_subset_train_test_suffix(rand_test_split_fract):
    rnd_train_suffix = f'RndTrain{(1-rand_test_split_fract)}'
    
    rnd_test_suffix = f'RndTest{rand_test_split_fract}'
    
    return rnd_train_suffix,rnd_test_suffix

rnd_train_suffix,rnd_test_suffix = get_rand_subset_train_test_suffix(rand_test_split_fract)

def define_final_class_ds_train_test_subset_names(class_ds_name_to_train_subset_name,rand_test_split_fract,class_ds_name_to_test_subset_names):
    final_class_ds_name_to_train_subset_name = {}
    final_class_ds_name_to_test_subset_names = {}
    
    rnd_train_suffix,rnd_test_suffix = get_rand_subset_train_test_suffix(rand_test_split_fract)
    
    for ds_name in class_ds_name_to_train_subset_name.keys():
        orig_train_subset_name = class_ds_name_to_train_subset_name[ds_name]
        
        final_rand_train_subset_name = f'{orig_train_subset_name}_FPs_{rnd_train_suffix}'
        final_rand_test_subset_name = f'{orig_train_subset_name}_FPs_{rnd_test_suffix}'
        
        final_class_ds_name_to_train_subset_name[ds_name] = final_rand_train_subset_name
        
        final_class_ds_name_to_test_subset_names[ds_name] = [f'{orig_test_subset_name}_FPs' for orig_test_subset_name in class_ds_name_to_test_subset_names[ds_name]]+[final_rand_test_subset_name]
    
    return final_class_ds_name_to_train_subset_name,final_class_ds_name_to_test_subset_names

def define_all_train_test_split_labels(class_ds_name_to_train_subset_name,rand_test_split_fract,class_ds_name_to_test_subset_names,ds_matched_to_ep_list,wang_test_set_names,no_wang_outer_folds):
    
    ds_matched_to_ep_to_all_train_test_pairs = doubleDefaultDictOfLists()
    
    final_class_ds_name_to_train_subset_name,final_class_ds_name_to_test_subset_names = define_final_class_ds_train_test_subset_names(class_ds_name_to_train_subset_name,rand_test_split_fract,class_ds_name_to_test_subset_names)
    
    for dataset_name in ds_matched_to_ep_list.keys():
        if 'classification' == get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names):
            for ep in ds_matched_to_ep_list[dataset_name]:
                print(f'Adding train:test pair labels for {dataset_name}:{ep}')
                train_set_label = f'{ep}_{final_class_ds_name_to_train_subset_name[dataset_name]}'
                
                for test_subset_name in final_class_ds_name_to_test_subset_names[dataset_name]:
                    test_set_label = f'{ep}_{test_subset_name}'
                    
                    ds_matched_to_ep_to_all_train_test_pairs[dataset_name][ep].append((train_set_label,test_set_label))
                
        elif 'regression' == get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names):
            for ep in ds_matched_to_ep_list[dataset_name]:
                for test_type_name in wang_test_set_names:
                    for fold in range(0,no_wang_outer_folds):
                        train_set_label = f'{ep}_{test_type_name}_f={fold}_train_FPs'
                        
                        test_set_label = f'{ep}_{test_type_name}_f={fold}_test_FPs'
                        
                        ds_matched_to_ep_to_all_train_test_pairs[dataset_name][ep].append((train_set_label,test_set_label))
        else:
            raise Exception(f'Unrecognised type of dataset: name = {dataset_name}, type = {get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)}')
    
    
    return ds_matched_to_ep_to_all_train_test_pairs

ds_matched_to_ep_to_all_train_test_pairs = define_all_train_test_split_labels(class_ds_name_to_train_subset_name,rand_test_split_fract,class_ds_name_to_test_subset_names,ds_matched_to_ep_list,wang_test_set_names,no_wang_outer_folds)

def check_the_same_test_set_label_is_never_seen_twice(ds_matched_to_ep_to_all_train_test_pairs):
    for dataset_name in ds_matched_to_exemplar_ep_list.keys():
        for endpoint in ds_matched_to_exemplar_ep_list[dataset_name]:
            
            all_test_set_labels = []
            
            for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
                
                train_set_label = train_test_label_pair[0]
                test_set_label = train_test_label_pair[1]
                
                all_test_set_labels.append(test_set_label)
            
            assert len(all_test_set_labels)==len(set(all_test_set_labels)),f'These test set labels are paired with different training set labels: {findDups(all_test_set_labels)}'

check_the_same_test_set_label_is_never_seen_twice(ds_matched_to_ep_to_all_train_test_pairs)

def reset_all_indices(fps_train,  fps_test, X_train_and_ids_df,  X_test_and_ids_df, train_y, test_y):
    #Random splitting will mean indices are not in the same order as the corresponding dataset IDs lists!
    #--------------------------------------
    all_inputs_df_status = [isinstance(df,pd.DataFrame) for df in [fps_train,  fps_test, X_train_and_ids_df,  X_test_and_ids_df]]
    if not all(all_inputs_df_status): raise Exception(f'all_inputs_df_status={all_inputs_df_status}')
    #---------------------------------------
    fps_train.reset_index(drop=True,inplace=True)
    
    fps_test.reset_index(drop=True,inplace=True)
    
    X_train_and_ids_df.reset_index(drop=True,inplace=True)
    
    X_test_and_ids_df.reset_index(drop=True,inplace=True)

    train_y.reset_index(drop=True,inplace=True)

    test_y.reset_index(drop=True,inplace=True)

def check_model_ready_train_or_test_subset_counts_are_internally_consistent(fps,ids,X_and_ids_df,y):
    assert isinstance(X_and_ids_df,pd.DataFrame)
    assert isinstance(y,pd.Series)
    assert isinstance(ids,list)
    assert isinstance(fps,pd.DataFrame)
    
    assert X_and_ids_df.shape[0] == fps.shape[0]
    assert X_and_ids_df.shape[0] == len(ids)
    assert X_and_ids_df.shape[0] == len(y)

def check_ready_for_modelling_and_ad_calculations(fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids):
    
    assert isinstance(X_train_and_ids_df,pd.DataFrame)
    assert isinstance(X_test_and_ids_df,pd.DataFrame)
    assert isinstance(train_y,pd.Series)
    assert isinstance(test_y,pd.Series)
    assert isinstance(test_ids,list)
    assert isinstance(train_ids,list)
    assert isinstance(fps_train,pd.DataFrame)
    assert isinstance(fps_test,pd.DataFrame)
    
    assert len(train_ids)==len(set(train_ids)),findDups(train_ids)
    assert len(test_ids)==len(set(test_ids)),findDups(test_ids)
    assert 0 == len(set(train_ids).intersection(set(test_ids)))
    
    check_fps_df(fps_train)
    check_fps_df(fps_test)
    
    assert list(range(X_train_and_ids_df.shape[0]))==X_train_and_ids_df.index.tolist()
    assert list(range(X_test_and_ids_df.shape[0]))==X_test_and_ids_df.index.tolist()
    assert train_y.index.tolist()==X_train_and_ids_df.index.tolist()
    assert test_y.index.tolist()==X_test_and_ids_df.index.tolist()
    assert fps_train.index.tolist()==X_train_and_ids_df.index.tolist()
    assert fps_test.index.tolist()==X_test_and_ids_df.index.tolist()
    
    check_model_ready_train_or_test_subset_counts_are_internally_consistent(fps=fps_train,ids=train_ids,X_and_ids_df=X_train_and_ids_df,y=train_y)
    assert X_train_and_ids_df[consistent_id_col].tolist() == train_ids
    
    check_model_ready_train_or_test_subset_counts_are_internally_consistent(fps=fps_test,ids=test_ids,X_and_ids_df=X_test_and_ids_df,y=test_y)
    assert X_test_and_ids_df[consistent_id_col].tolist() == test_ids
    
    

def load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,  train_set_label,  test_set_label):
    out_train_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{train_set_label}.pkl'])
    
    out_test_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{test_set_label}.pkl'])
    
    try:
        train_dict = load_from_pkl_file(out_train_pkl_file)
        
        fps_train = train_dict['fps_train']
        X_train_and_ids_df = train_dict['X_train_and_ids_df']
        train_y = train_dict['train_y']
        train_ids = train_dict['train_ids']
    except Exception as err:
        raise Exception(f'Problem reading from out_train_pkl_file={out_train_pkl_file} :{type(err)}:{str(err)}')
    
    try:
        test_dict = load_from_pkl_file(out_test_pkl_file)
        
        fps_test = test_dict['fps_test']
        X_test_and_ids_df = test_dict['X_test_and_ids_df']
        test_y = test_dict['test_y']
        test_ids = test_dict['test_ids']
    except Exception as err:
        raise Exception(f'Problem reading from out_test_pkl_file={out_test_pkl_file} :{type(err)}:{str(err)}')
    
    
    reset_all_indices(fps_train,  fps_test, X_train_and_ids_df,  X_test_and_ids_df, train_y, test_y)
    
    check_ready_for_modelling_and_ad_calculations(fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids)
    
    return fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids

def populateDictWithStats(dict_of_stats,dict_of_current_stats,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset):
    
    for stat_name in dict_of_current_stats.keys():
        dict_of_stats[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset][stat_name] = dict_of_current_stats[stat_name]

def writeStatsFiles(statsFilesDict,dataset_name,dict_of_stats,calc_dir,context_label,endpoint,test_set_label,rand_seed,AD_method_name,method):
    #----------------------------
    assert context_label == f't={test_set_label}_s={rand_seed}_ad={AD_method_name}_m={method}',f'Unexpected context_label={context_label}'
    assert endpoint in test_set_label,f'We need endpoint={endpoint} to be present in test_set_label={test_set_label} if endpoint is not explicitly included in context_label!'
    #----------------------------


              
    dd = dict_of_stats[endpoint][test_set_label][rand_seed][AD_method_name][method]
    
    df = convertDefaultDictDictIntoDataFrame(dd,col_name_for_first_key='AD Subset')
                        
    stats_file = os.path.sep.join([calc_dir,f'{context_label}_Statistics.csv'])
                        
    statsFilesDict[dataset_name][endpoint][test_set_label][rand_seed][AD_method_name][method] = stats_file
                        
    df.to_csv(stats_file,index=False)
    
    return statsFilesDict

def assign_id_corresponding_to_inchi_to_one_row(row,consistent_id_col,consistent_inchi_col,inchi2newId):
    
    inchi = row[consistent_inchi_col]
    
    new_id = inchi2newId[inchi]
    
    row[consistent_id_col] = new_id

    return row

def assign_new_ids_corresponding_to_inchis(unique_df, consistent_inchi_col, consistent_id_col):
    all_inchis = unique_df[consistent_inchi_col].tolist()
    
    assert len(all_inchis)==len(set(all_inchis))
    
    all_inchis.sort()
    
    inchi2newId = dict(zip(all_inchis,list(range(len(all_inchis)))))
    
    unique_df=unique_df.apply(assign_id_corresponding_to_inchi_to_one_row,axis=1,args=(consistent_id_col,consistent_inchi_col,inchi2newId))
    
    return unique_df

def get_ds_matched_to_other_endpoints_of_interest(ds_matched_to_ep_list,ds_matched_to_exemplar_ep_list):

    ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list = defaultdict(list)

    for ds in ds_matched_to_ep_list.keys():
        ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list[ds] = [ep for ep in ds_matched_to_ep_list[ds] if not ep in ds_matched_to_exemplar_ep_list[ds]]

    return ds_matched_to_other_endpoints_not_in_exemplar_endpoints_list

