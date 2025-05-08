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
#Copyright (c) 2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys,glob,re,pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit 
#-----------------
from common_globals import consistent_id_col,consistent_act_col,consistent_smiles_col,consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,pkg_dir,top_class_or_reg_ds_dirs,regression_dataset_names,classification_dataset_names,ds_matched_to_ep_list,get_endpoint_type
from common_globals import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_set_names,no_wang_outer_folds
from common_globals import all_Tox21_endpoints,all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest,tox21_subsets_of_interest
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,fp_col,stand_mol_inchi_col,stand_mol_smiles_col,subset_name_col
from common_globals import get_rand_subset_train_test_suffix,rand_split_seed,no_train_test_splits,rand_test_split_fract
from common_globals import act_col_to_use_for_filtering_based_upon_stand_mol_inchis
#------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.CheminformaticsUtils.standardize_mols import standardize
from modelsADuncertaintyPkg.utils.basic_utils import check_no_missing_values
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import findDups
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,create_pkl_file
from common_globals import assign_new_ids_corresponding_to_inchis
#==========================================


def combine_all_subsets_again(dataset_name,endpoint_type,dir_with_files_to_parse,endpoint,subset_name_col,consistent_id_col,wang_test_name,wang_fold,chembl_subsets_of_interest,tox21_subsets_of_interest):
    
    if 'classification' == endpoint_type:
        if 'Morger_Tox21' == dataset_name:
            subsets_of_interest = [f'{endpoint}_{subset_type}' for subset_type in tox21_subsets_of_interest]
        elif 'Morger_ChEMBL' == dataset_name:
            subsets_of_interest = [f'{endpoint}_{subset_type}' for subset_type in chembl_subsets_of_interest]
        else:
            raise Exception(f'Unrecognized dataset name={dataset_name}')
    elif 'regression' == endpoint_type:
        assert 'Wang_ChEMBL' == dataset_name,dataset_name
        
        subsets_of_interest = [f'{endpoint}_{wang_test_name}_f={wang_fold}_{subset_suffix}' for subset_suffix in ['train','test']]
    else:
        raise Exception(f'Dataset={dataset_name}. Unrecognised endpoint_type={endpoint_type}')
    
    files_of_interest = [os.path.sep.join([f'{dir_with_files_to_parse}',f'{subset}_NRsFilt.csv']) for subset in subsets_of_interest]
    
    all_subset_dfs = []
    
    for file in files_of_interest:
        try:
            all_subset_dfs.append(pd.read_csv(file))
        except Exception as err:
            raise Exception(f'{err} - Problem reading {file}')
    
    combined_df = pd.concat(all_subset_dfs)
    
    return combined_df

def add_stand_mol_col(combined_df,consistent_smiles_col,stand_mol_col):
    combined_df.insert(combined_df.shape[1],stand_mol_col,[standardize(smiles) for smiles in combined_df[consistent_smiles_col].tolist()],allow_duplicates=True)
    
    combined_standardized_mols_df = combined_df
    
    return combined_standardized_mols_df

def add_stand_mol_smiles_col(combined_standardized_mols_df,stand_mol_col,stand_mol_smiles_col):
    combined_standardized_mols_df.insert(combined_standardized_mols_df.shape[1],stand_mol_smiles_col,[ChemDataParser.Chem.MolToSmiles(mol) for mol in combined_standardized_mols_df[stand_mol_col].tolist()],allow_duplicates=True)
    
    combined_standardized_mols_with_smiles_df = combined_standardized_mols_df
    
    
    return combined_standardized_mols_with_smiles_df

def add_act_col_to_use_for_filtering_based_upon_stand_mol_inchis(combined_standardized_mols_df,act_col_to_use_for_filtering_based_upon_stand_mol_inchis,consistent_act_col,endpoint_type):
    
    if 'classification' == endpoint_type:
        list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis = combined_standardized_mols_df[consistent_act_col].tolist()
    elif 'regression' == endpoint_type:
        list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis = list(range(0,combined_standardized_mols_df.shape[0]))
        assert len(list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis)==len(set(list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis)),findDups(list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis)
    else:
        raise Exception(f'Unrecognised endpoint_type={endpoint_type}')
    
    combined_standardized_mols_df.insert(combined_standardized_mols_df.shape[1],act_col_to_use_for_filtering_based_upon_stand_mol_inchis,list_of_act_vals_to_use_for_filtering_based_on_stand_mol_inchis,allow_duplicates=True)
    
    combined_ready_to_filter_df = combined_standardized_mols_df
    
    return combined_ready_to_filter_df

def add_default_fingerprints_from_stand_mols(unique_df,stand_mol_col,fp_col,consistent_id_col):
    
    unique_df,fps_df = ChemDataParser.getFpsDfAndDataDfSkippingFailedMols(data_df=unique_df,smiles_col=None,id_col=consistent_id_col,fp_col=fp_col,pre_calc_stand_mol_col=stand_mol_col)
    
    unique_df.insert(0,fp_col,fps_df[fp_col].tolist(),allow_duplicates=True)
    
    unique_with_fps_df = unique_df
    
    return unique_with_fps_df

def write_filtered_subset_pickle_files(endpoint,subset_name_col,unique_with_final_cols_df,out_dir):
    
    unique_subset_names = unique_with_final_cols_df[subset_name_col].unique().tolist()
    
    for subset_name in unique_subset_names:
        subset_pickle_file = os.path.sep.join([out_dir,f'{endpoint}_{subset_name}_FPs.pkl'])
        
        subset_df = unique_with_final_cols_df[unique_with_final_cols_df[subset_name_col].isin([subset_name])]
        
        subset_df.reset_index(drop=True, inplace=True)
        
        create_pkl_file(pkl_file=subset_pickle_file,obj=subset_df)
        
def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        endpoint_type = get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter2'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter3'])
        
        assert not out_dir == dir_with_files_to_parse,dir_with_files_to_parse
        
        createOrReplaceDir(dir_=out_dir)
        
        for endpoint in ds_matched_to_ep_list[dataset_name]:
            for wang_test_name in wang_test_set_names:
                for wang_fold in range(0,no_wang_outer_folds):
                    #------------------------------------------
                    if not 'Wang_ChEMBL' == dataset_name:
                        if not (wang_test_name == wang_test_set_names[0] and wang_fold == 0):
                            continue
                    #-------------------------------------------
                    
                    combined_df = combine_all_subsets_again(dataset_name,endpoint_type,dir_with_files_to_parse,endpoint,subset_name_col,consistent_id_col,wang_test_name,wang_fold,chembl_subsets_of_interest,tox21_subsets_of_interest)
                    
                    combined_standardized_mols_df = add_stand_mol_col(combined_df,consistent_smiles_col,stand_mol_col)
                    
                    combined_standardized_mols_with_smiles_df = add_stand_mol_smiles_col(combined_standardized_mols_df,stand_mol_col,stand_mol_smiles_col)
                    
                    combined_ready_to_filter_df = add_act_col_to_use_for_filtering_based_upon_stand_mol_inchis(combined_standardized_mols_df,act_col_to_use_for_filtering_based_upon_stand_mol_inchis,consistent_act_col,endpoint_type)
                    
                    ###################################
                    #For classification datasets, we could remove compounds with duplicate InChIs from standardized molecules and inconsistent class labels, or just keep one example if all class labels are consistent, as per "Data preprocessing" in Morger et al. (2021) [https://link.springer.com/article/10.1186/s13321-021-00511-5]
                    #However, for regression datasets, it would not be appropriate to average activity values assigned to these duplicate InChIs from standardized molecules, as these represent different compounds. Hence, we just treat these as classification datasets, after assigning a column of unique "class labels" [see add_act_col_to_use_for_filtering_based_upon_stand_mol_inchis(..._)], i.e. any compounds which are the same after standardizing are just dropped completely!
                    
                    
                    unique_df = ChemDataParser.removeDuplicateMolsTakingAccountOfActivities(df=combined_ready_to_filter_df,smiles_col=stand_mol_smiles_col,activity_col=act_col_to_use_for_filtering_based_upon_stand_mol_inchis,unique_mol_id_col=None,type_of_activities="classification",drop_intermediate_cols=False)
                    
                    unique_df = unique_df.rename({'InChI':stand_mol_inchi_col},axis=1)
                    ######################################
                    
                    #############################
                    #This is needed because filtering based on InChIs could have inconsistently dropped rows corresponding to a different arbitary IDs for different train/test splits of the Wang-ChEMBL dataset. However, the IDs need to be consistent across different train/splits for summarize_dataset_stats_before_and_after_parsing.py:
                    if 'Wang_ChEMBL' == dataset_name:
                        unique_df = assign_new_ids_corresponding_to_inchis(unique_df, stand_mol_inchi_col, consistent_id_col)
                    #############################
                    
                    unique_with_fps_df = add_default_fingerprints_from_stand_mols(unique_df,stand_mol_col,fp_col,consistent_id_col)
                    
                    unique_with_final_cols_df = unique_with_fps_df[[subset_name_col,consistent_id_col,consistent_act_col,consistent_smiles_col,consistent_inchi_col,stand_mol_col,stand_mol_smiles_col,stand_mol_inchi_col,fp_col]]
                    
                    check_no_missing_values(unique_with_final_cols_df)
                    
                    write_filtered_subset_pickle_files(endpoint,subset_name_col,unique_with_final_cols_df,out_dir)
                    
                    #============================
                    #if not 'Wang_ChEMBL' == dataset_name: #We already have random splits from Wang et al. (2021)
                    #    create_rand_train_test_split_pickles(out_dir,endpoint,dataset_name,consistent_act_col,class_ds_name_to_train_subset_name,rand_test_split_fract,rand_split_seed,no_train_test_splits,consistent_id_col)
                    #=============================
                    
        #convert_final_pickles_to_csv_for_quick_checking(out_dir)
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
