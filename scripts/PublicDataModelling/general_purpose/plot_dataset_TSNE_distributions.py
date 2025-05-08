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
import os,   sys,   time
from collections import defaultdict
import pandas as pd
import numpy as np
#=============================
from common_globals import consistent_id_col
from common_globals import pkg_dir,   top_class_or_reg_ds_dirs,   ds_matched_to_ep_list
from common_globals import load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations
from common_globals import ds_matched_to_ep_to_all_train_test_pairs
from common_globals import rand_test_split_fract
##################################
#from define_plot_settings import *
##################################


dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
#----------------------------------------------------------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.CheminformaticsUtils.visualize_chemical_space import get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels
from modelsADuncertaintyPkg.CheminformaticsUtils.visualize_chemical_space import prepare_X_and_ids_df_for_TSNE
##################################

def this_is_an_ivit_train_or_test_label(train_or_test_label):
    if 'ivit' in train_or_test_label:
        return True
    elif 'ivot' in train_or_test_label:
        return False
    else:
        raise Exception(train_or_test_label)

def tidy_up_subset_label(dataset_name,train_or_test_label,rand_test_split_fract=rand_test_split_fract):
    if 'Wang_ChEMBL' == dataset_name:
        
        if 'train' in train_or_test_label:
            label_suffix = '(train)'
        elif 'test' in train_or_test_label:
            label_suffix = '(test)'
        else:
            raise Exception(train_or_test_label)
        
        fold = int(train_or_test_label.split('f=')[1].split('_')[0])

        label = f'fold-{fold} {label_suffix}'
    elif 'Morger_ChEMBL' == dataset_name or 'Morger_Tox21' == dataset_name:
        
        if 'Morger_ChEMBL' == dataset_name:
            expected_subsets = ['update1','holdout',f'RndTrain{(1-rand_test_split_fract)}',f'RndTest{rand_test_split_fract}']
        elif 'Morger_Tox21' == dataset_name:
            expected_subsets = ['Tox21Score','Tox21Test',f'RndTrain{(1-rand_test_split_fract)}',f'RndTest{rand_test_split_fract}']
        else:
            raise Exception(dataset_name)
        
        found_subset = False
        for expected_subset in expected_subsets:
            if expected_subset in train_or_test_label:
                found_subset = True

                if 'RndTrain' in expected_subset:
                    label = 'Train'
                elif 'RndTest' in expected_subset:
                    label = 'Random'
                else:
                    label = expected_subset
        
        if not found_subset: raise Exception(train_or_test_label)
    
    else:
        raise Exception(dataset_name)
    
    return label

def  get_all_TSNE_plots(dataset_name,  dir_with_files_to_parse,   out_dir,   ds_matched_to_relevant_ep_list,   ds_matched_to_ep_to_all_train_test_pairs,id_col,subset_col="Subset"):
    
    for endpoint in ds_matched_to_relevant_ep_list[dataset_name]:

        ep_start = time.time()

        subsets_to_include_in_tsne_dict = defaultdict(list)

        seen_train_labels = []
        
        for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[dataset_name][endpoint]:
            
            start = time.time()
            
            train_set_label = train_test_label_pair[0]
            test_set_label = train_test_label_pair[1]
            
            fps_train,   fps_test,   test_ids,   X_train_and_ids_df,   X_test_and_ids_df,   train_y,   test_y,   train_ids =load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,   train_set_label,   test_set_label)
            
            end = time.time()
            
            print('='*20)
            print('Loaded fingerprints for:')
            print(f'dataset_name={dataset_name}, endpoint={endpoint}')
            print(f'train_test_label_pair={train_test_label_pair}')
            print(f'Time taken = {(end-start)/60} minutes')
            print('='*20)

            orig_train_set_label = train_set_label
            orig_test_set_label = test_set_label
            train_set_label = tidy_up_subset_label(dataset_name,train_or_test_label=train_set_label)
            test_set_label = tidy_up_subset_label(dataset_name,train_or_test_label=test_set_label)
            
            test_df_ready_for_tsne = prepare_X_and_ids_df_for_TSNE(X_and_ids_df=X_test_and_ids_df,subset_col=subset_col,subset_name=test_set_label,id_col=id_col)

            if not 'Wang_ChEMBL' == dataset_name:

                subsets_to_include_in_tsne_dict['all'].append(test_df_ready_for_tsne)

                if not train_set_label in seen_train_labels:
                    train_df_ready_for_tsne = prepare_X_and_ids_df_for_TSNE(X_and_ids_df=X_train_and_ids_df,subset_col=subset_col,subset_name=train_set_label,id_col=id_col)
            
                    subsets_to_include_in_tsne_dict['all'].append(train_df_ready_for_tsne)
            else:
                if this_is_an_ivit_train_or_test_label(train_or_test_label=orig_test_set_label) and this_is_an_ivit_train_or_test_label(train_or_test_label=orig_train_set_label):
                    subset_group = 'IVIT'
                elif not this_is_an_ivit_train_or_test_label(train_or_test_label=orig_test_set_label) and not this_is_an_ivit_train_or_test_label(train_or_test_label=orig_train_set_label):
                    subset_group = 'IVOT'
                else:
                    raise Exception(f'orig_test_set_label={orig_test_set_label},orig_train_set_label={orig_train_set_label}')
                
                subsets_to_include_in_tsne_dict[subset_group].append(test_df_ready_for_tsne)

        for subset_group in subsets_to_include_in_tsne_dict.keys():

            all_subsets_df = pd.concat(subsets_to_include_in_tsne_dict[subset_group])

            all_subsets_df.reset_index(drop=True,inplace=True)

            title_prefix = f'{endpoint}_{subset_group}'

            plot_name_prefix = os.path.sep.join([out_dir,title_prefix])

            get_tsne_plot_of_dataset_with_fp_bit_vector_array_and_subset_labels(plot_name_prefix,title_prefix,dataset_df=all_subsets_df,subset_col=subset_col)

        ep_end = time.time()

        print('='*20)
        print('Computed TSNE plots for:')
        print(f'dataset_name={dataset_name}, endpoint={endpoint}')
        print(f'Time taken = {(ep_end-ep_start)/60} minutes')
        print('='*20)
            
            
            
    
    


def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        
        dir_with_files_to_parse = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'Model_Ready'])
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],   'TSNE'])
        
        assert not out_dir == dir_with_files_to_parse,   dir_with_files_to_parse
        
        createOrReplaceDir(dir_=out_dir)
        
        get_all_TSNE_plots(dataset_name,  dir_with_files_to_parse,   out_dir,   ds_matched_to_relevant_ep_list=ds_matched_to_ep_list,   ds_matched_to_ep_to_all_train_test_pairs=ds_matched_to_ep_to_all_train_test_pairs,id_col=consistent_id_col)
        
        
        
        
            
            
        
        
        
        
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

