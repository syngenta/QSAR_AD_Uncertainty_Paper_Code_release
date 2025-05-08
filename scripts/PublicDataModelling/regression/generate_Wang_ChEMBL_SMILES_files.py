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
from collections import defaultdict
import pandas as pd
import numpy as np
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#--------------------------------------------
from common_globals_regression_scripts import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,no_wang_outer_folds,wang_test_set_names,wang_ids_col,wang_top_ds_dir
#--------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import findDups
#--------------------------------------------
plus_ids_suffix = '_PlusIDs.csv'
top_ds_dir = wang_top_ds_dir

def add_unique_ids_to_all_compounds(top_ds_dir,all_Wang_endpoints,plus_ids_suffix,wang_ids_col):
    for endpoint in all_Wang_endpoints:
        raw_csv = os.path.sep.join([top_ds_dir,'data','dataset',f'{endpoint}.csv'])
        
        ###########################
        new_csv = re.sub('(\.csv$)',plus_ids_suffix,raw_csv)
        
        assert not raw_csv == new_csv,raw_csv
        ###########################
        
        df = pd.read_csv(raw_csv)
        
        df.insert(0,wang_ids_col,range(0,df.shape[0]))#,allow_duplicates=True)
        
        df.to_csv(new_csv,index=False)
    

def load_split_info(file_with_fold_to_wang_train_val_test_indices):
    f_in = open(file_with_fold_to_wang_train_val_test_indices,'rb')
    try:
        split_info = pickle.load(f_in)
    finally:
        f_in.close()
    
    return split_info

def get_fold_to_train_test_indices(split_info,no_wang_outer_folds,endpoint,test_name):
    fold_to_train_test_indices = defaultdict(dict)
    
    for fold in range(0,no_wang_outer_folds):
        try:
            fold_specific_info = split_info[fold]
        except IndexError as err:
            raise Exception(f'err={err},endpoint={endpoint},test_name={test_name},fold={fold}')
        
        wang_train,wang_val,wang_test = fold_specific_info #I tried this in the interpreter with one example and got a ValueError if too many or too few 'values to unpack'
        
        fold_to_train_test_indices[fold]['train'] = wang_train+wang_val
        
        fold_to_train_test_indices[fold]['test'] = wang_test
    
    return fold_to_train_test_indices

def check_consistency_of_indices_used_for_splits_and_all_ids(split_ready_df,ids_col,endpoint,test_name,fold,combined_indices):
    combined_indices.sort()
    
    assert len(combined_indices)==len(set(combined_indices)),f"endpoint={endpoint},test_name={test_name},fold={fold}. Duplicate indices={findDups(combined_indices)}"
    
    all_ids = split_ready_df[ids_col].tolist()
    
    all_ids.sort()
    
    assert combined_indices == all_ids,f"endpoint={endpoint},test_name={test_name},fold={fold}. Indices in combined_indices not present in all_ids={[index for index in combined_indices if not index in all_ids]}. IDs in all_ids not present in combined_indices={[id_ for id_ in all_ids if not id_ in combined_indices]}"

def check_fold_to_train_test_indices(fold_to_train_test_indices,split_ready_df,endpoint,test_name,test_perc_expected=20,tolerance=2,ids_col=wang_ids_col):
    all_test_indices = []
    
    for fold in fold_to_train_test_indices.keys():
        train_indices = fold_to_train_test_indices[fold]['train']
        
        test_indices = fold_to_train_test_indices[fold]['test']
        
        assert all([isinstance(i,int) for i in train_indices]),f"endpoint={endpoint},test_name={test_name},fold={fold},train_indices unique types={list(set([type(i) for i in train_indices]))}"
        
        assert all([isinstance(i,int) for i in test_indices]),f"endpoint={endpoint},test_name={test_name},fold={fold},test_indices unique types={list(set([type(i) for i in test_indices]))}"
        
        test_perc=100*(len(test_indices)/(len(test_indices)+len(train_indices)))
        
        fold_info_str = f"endpoint={endpoint},test_name={test_name},fold={fold},test_perc={test_perc}"
        
        assert abs(test_perc-test_perc_expected) <= tolerance,fold_info_str
        
        print(fold_info_str)
        
        all_test_indices += test_indices
        
        combined_indices = train_indices[:]+test_indices[:]
        
        check_consistency_of_indices_used_for_splits_and_all_ids(split_ready_df,ids_col,endpoint,test_name,fold,combined_indices)
    
    assert len(all_test_indices)==len(set(all_test_indices)),f"endpoint={endpoint},test_name={test_name},duplicate test indices-{findDups}"
    
    assert len(all_test_indices)==split_ready_df.shape[0],f"endpoint={endpoint},test_name={test_name},len(all_test_indices)={len(all_test_indices)},split_ready_df.shape[0]={split_ready_df.shape[0]}"
    
    check_consistency_of_indices_used_for_splits_and_all_ids(split_ready_df,ids_col,endpoint,test_name,fold='N/A',combined_indices=all_test_indices)

def split_file(top_ds_dir,endpoint,test_name,fold,split_ready_df,fold_to_train_test_indices):
    for subset in ['train','test']:
        split_file_name = os.path.sep.join([top_ds_dir,'data','dataset',f'{endpoint}_{test_name}_f={fold}_{subset}.csv'])
        
        row_integer_indices = fold_to_train_test_indices[fold][subset]
        
        subset_df = split_ready_df.iloc[row_integer_indices]
        
        subset_df.to_csv(split_file_name,index=False)

def split_into_train_test_subsets(top_ds_dir,all_Wang_endpoints,wang_test_set_names,plus_ids_suffix,no_wang_outer_folds):
    for endpoint in all_Wang_endpoints:
        
        split_ready_csv = os.path.sep.join([top_ds_dir,'data','dataset',f'{endpoint}{plus_ids_suffix}'])
        
        split_ready_df = pd.read_csv(split_ready_csv)
        
        for test_name in wang_test_set_names:
            file_with_fold_to_wang_train_val_test_indices = os.path.sep.join([top_ds_dir,'data',test_name,f'{endpoint}_{test_name}.pkl'])
            
            split_info = load_split_info(file_with_fold_to_wang_train_val_test_indices)
            
            fold_to_train_test_indices = get_fold_to_train_test_indices(split_info,no_wang_outer_folds,endpoint,test_name)
            
            check_fold_to_train_test_indices(fold_to_train_test_indices,split_ready_df,endpoint,test_name)
            
            for fold in fold_to_train_test_indices.keys():
                
                split_file(top_ds_dir,endpoint,test_name,fold,split_ready_df,fold_to_train_test_indices)

def main():
    print('THE START')
    
    add_unique_ids_to_all_compounds(top_ds_dir,all_Wang_endpoints,plus_ids_suffix,wang_ids_col)
    
    split_into_train_test_subsets(top_ds_dir,all_Wang_endpoints,wang_test_set_names,plus_ids_suffix,no_wang_outer_folds) #We use Wang [train+validation] for training and, where applicable, calibration. We use the test subsets as the test sets.
    
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

