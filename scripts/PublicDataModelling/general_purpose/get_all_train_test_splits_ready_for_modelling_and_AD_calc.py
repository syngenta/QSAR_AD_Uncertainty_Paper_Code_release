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
#-----------------
from common_globals import consistent_id_col,consistent_act_col,consistent_smiles_col,consistent_inchi_col
from common_globals import top_dir_of_public_data_scripts,pkg_dir,top_class_or_reg_ds_dirs,regression_dataset_names,classification_dataset_names,ds_matched_to_ep_list,get_endpoint_type
from common_globals import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_set_names,no_wang_outer_folds
from common_globals import all_Tox21_endpoints,all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest,tox21_subsets_of_interest
from common_globals import class_ds_name_to_train_subset_name
from common_globals import stand_mol_col,fp_col,stand_mol_inchi_col,stand_mol_smiles_col,subset_name_col
from common_globals import get_rand_subset_train_test_suffix,rand_split_seed,no_train_test_splits
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
from common_globals import act_col_to_use_for_filtering_based_upon_stand_mol_inchis
#-----------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.utils.basic_utils import check_no_missing_values
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,create_pkl_file
#==========================================

def load_outputs_from_last_filter_script(dir_with_files_to_parse,train_set_label,test_set_label,cols_to_drop):
    train_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{train_set_label}.pkl'])
    
    test_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{test_set_label}.pkl'])
    
    train_df = load_from_pkl_file(train_pkl_file).drop(labels=cols_to_drop,axis=1) 
    
    test_df = load_from_pkl_file(test_pkl_file).drop(labels=cols_to_drop,axis=1)
    
    return train_df,test_df

def create_pkl_files_ready_for_modelling_and_AD_calculations(out_dir,train_set_label,test_set_label,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids):
    out_train_pkl_file = os.path.sep.join([out_dir,f'{train_set_label}.pkl'])
    
    out_test_pkl_file = os.path.sep.join([out_dir,f'{test_set_label}.pkl'])
    
    train_dict = {}
    train_dict['fps_train'] = fps_train
    train_dict['X_train_and_ids_df'] = X_train_and_ids_df
    train_dict['train_y'] = train_y
    train_dict['train_ids'] = train_ids
    
    test_dict = {}
    test_dict['fps_test'] = fps_test
    test_dict['X_test_and_ids_df'] = X_test_and_ids_df
    test_dict['test_y'] = test_y
    test_dict['test_ids'] = test_ids
    
    create_pkl_file(out_train_pkl_file,train_dict)
    
    create_pkl_file(out_test_pkl_file,test_dict)

def main():
    print('THE START')
    
    cols_to_drop = [consistent_inchi_col,stand_mol_inchi_col,stand_mol_smiles_col,subset_name_col] #act_col_to_use_for_filtering_based_upon_stand_mol_inchis was filtered shortly after being added!
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter3'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Model_Ready'])
        
        assert not out_dir == dir_with_files_to_parse,dir_with_files_to_parse
        
        createOrReplaceDir(dir_=out_dir)
        
        for endpoint in ds_matched_to_ep_to_all_train_test_pairs[dataset_name]:
            for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
                
                train_set_label = train_test_label_pair[0]
                test_set_label = train_test_label_pair[1]
                
                train_df,test_df = load_outputs_from_last_filter_script(dir_with_files_to_parse,train_set_label,test_set_label,cols_to_drop)
                
                fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids = getInputRequiredForModellingAndAD(train_df,test_df,id_col=consistent_id_col,smiles_col=consistent_smiles_col,class_label_col=consistent_act_col,class_1=None,fp_col=fp_col,pre_calc_stand_mol_col=stand_mol_col,pre_calc_fps=True) #class_1 is actually a redudndant argument here! #usage of this function double-checked against test_getInputRequiredForModellingAndAD_from_smiles_with_pre_calc_default_fps(...)
                
                create_pkl_files_ready_for_modelling_and_AD_calculations(out_dir,train_set_label,test_set_label,fps_train,fps_test,test_ids,X_train_and_ids_df,X_test_and_ids_df,train_y,test_y,train_ids)
    
    
    print('THE END')

if __name__ == '__main__':
    sys.exit(main())
