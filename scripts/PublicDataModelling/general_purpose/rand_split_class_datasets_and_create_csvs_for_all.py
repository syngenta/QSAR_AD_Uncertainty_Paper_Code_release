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
#==========================================


def load_class_train_subset(out_dir,endpoint,dataset_name,class_ds_name_to_train_subset_name):
    subset_name = class_ds_name_to_train_subset_name[dataset_name]
    
    subset_pickle_file = os.path.sep.join([out_dir,f'{endpoint}_{subset_name}_FPs.pkl'])
    
    subset_df = load_from_pkl_file(pkl_file=subset_pickle_file)
    
    return subset_df,subset_pickle_file

def write_rnd_subset_pkl_file(rnd_subset_df,rnd_subset_suffix,orig_subset_pkl_file):
    new_pkl_file = re.sub('(\.pkl$)',f'_{rnd_subset_suffix}.pkl',orig_subset_pkl_file)
    assert not orig_subset_pkl_file == new_pkl_file,orig_subset_pkl_file
    
    create_pkl_file(pkl_file=new_pkl_file,obj=rnd_subset_df)
    

def create_rand_train_test_split_pickles(out_dir,endpoint,dataset_name,consistent_act_col,class_ds_name_to_train_subset_name,rand_test_split_fract,rand_split_seed,no_train_test_splits,consistent_id_col):
    
    subset_df,subset_pickle_file = load_class_train_subset(out_dir,endpoint,dataset_name,class_ds_name_to_train_subset_name)
    
    rnd_train_suffix,rnd_test_suffix = get_rand_subset_train_test_suffix(rand_test_split_fract)
    
    
    placeholder_for_x = np.zeros(subset_df.shape[0]) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit.split: can use a placeholder instead of X
    
    dataset_y = subset_df[consistent_act_col]
    
    assert isinstance(dataset_y,pd.Series) and dataset_y.shape[0] == subset_df.shape[0] # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit.split : dataset_y must be "array-like" #https://scikit-learn.org/stable/glossary.html: numeric pandas.Series is "array-like"
    
    for train_indices, test_indices in StratifiedShuffleSplit(n_splits=no_train_test_splits, random_state=rand_split_seed,test_size=rand_test_split_fract).split(placeholder_for_x,dataset_y):
        train_df = subset_df.iloc[train_indices,:]
        
        test_df = subset_df.iloc[test_indices,:]
        
        assert 0 == len(set(train_df[consistent_id_col].tolist()).intersection(set(test_df[consistent_id_col].tolist())))
        
        write_rnd_subset_pkl_file(rnd_subset_df=train_df,rnd_subset_suffix=rnd_train_suffix,orig_subset_pkl_file=subset_pickle_file)
        
        write_rnd_subset_pkl_file(rnd_subset_df=test_df,rnd_subset_suffix=rnd_test_suffix,orig_subset_pkl_file=subset_pickle_file)

def convert_final_pickles_to_csv_for_quick_checking(out_dir):
    for pkl_file in glob.glob(os.path.sep.join([out_dir,'*.pkl'])):
        contents_df = load_from_pkl_file(pkl_file)
        
        csv_file = re.sub('(\.pkl$)','.csv',pkl_file)
        assert not pkl_file == csv_file,pkl_file
        
        contents_df.to_csv(csv_file,index=True)

def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        endpoint_type = get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)
        
        #dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter2'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter3'])
        
        #assert not out_dir == dir_with_files_to_parse,dir_with_files_to_parse
        
        #createOrReplaceDir(dir_=out_dir) #This will have been created by the previous script!
        
        if dataset_name in classification_dataset_names:
            for endpoint in ds_matched_to_ep_list[dataset_name]:
                create_rand_train_test_split_pickles(out_dir,endpoint,dataset_name,consistent_act_col,class_ds_name_to_train_subset_name,rand_test_split_fract,rand_split_seed,no_train_test_splits,consistent_id_col)
        
        convert_final_pickles_to_csv_for_quick_checking(out_dir)
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
