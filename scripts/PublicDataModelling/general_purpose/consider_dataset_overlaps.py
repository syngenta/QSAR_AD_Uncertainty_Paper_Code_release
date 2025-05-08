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
#Copyright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys
import pandas as pd
from collections import defaultdict
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
from common_globals import overall_top_public_data_dir
from  common_globals import ds_matched_to_exemplar_ep_list
from common_globals import load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations
#-----------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import check_no_missing_values, neverEndingDefaultDict
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
from modelsADuncertaintyPkg.utils.basic_utils import flatten
from modelsADuncertaintyPkg.utils.basic_utils import report_name_of_function_where_this_is_called
from modelsADuncertaintyPkg.qsar_AD.applyADmethods import getInputRequiredForModellingAndAD
from modelsADuncertaintyPkg.utils.basic_utils import load_from_pkl_file,create_pkl_file,doubleDefaultDictOfLists
#==========================================

def load_train_test_subsets_from_pkls_without_dropping_cols(dir_with_files_to_parse,train_set_label,test_set_label):
        
    train_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{train_set_label}.pkl'])
    
    test_pkl_file = os.path.sep.join([dir_with_files_to_parse,f'{test_set_label}.pkl'])
    
    train_df = load_from_pkl_file(train_pkl_file)
    
    test_df = load_from_pkl_file(test_pkl_file)

    return train_df,test_df

def get_inchis_ids_tuples_list(df,inchi_col,id_col):
    inchis_list = df[inchi_col].tolist()

    ids_list = df[id_col].tolist()

    inchis_ids_tuples_list = [(inchis_list[i],ids_list[i]) for i in range(df.shape[0])]

    return inchis_ids_tuples_list


def check_inchis_and_ids_are_unique_across_train_test_combination(train_inchis_ids_tuples_list,test_inchis_ids_tuples_list):
        
    train_ids = [t[1] for t in train_inchis_ids_tuples_list]
    test_ids = [t[1] for t in test_inchis_ids_tuples_list]
    
    all_ids = train_ids+test_ids
    assert len(all_ids)==len(set(all_ids))

    train_inchis = [t[0] for t in train_inchis_ids_tuples_list]
    test_inchis = [t[0] for t in test_inchis_ids_tuples_list]

    all_inchis = train_inchis+test_inchis
    assert len(all_inchis)==len(set(all_inchis))


def get_dataset_to_endpoint_to_inchis_list(top_class_or_reg_ds_dirs,ds_matched_to_ep_to_all_train_test_pairs,subdir_with_pre_model_ready_datasets='Filter3',inchi_col=stand_mol_inchi_col,id_col=consistent_id_col):
    print(f'Calling this function: {report_name_of_function_where_this_is_called()}')

    ds_to_ep_to_inchis_ids_tuples_list = doubleDefaultDictOfLists()

    for ds in top_class_or_reg_ds_dirs.keys():
        top_dir = top_class_or_reg_ds_dirs[ds]

        dir_with_files_to_parse = os.path.join(top_dir,subdir_with_pre_model_ready_datasets)
        
        for endpoint in ds_matched_to_ep_to_all_train_test_pairs[ds]:
            for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[ds][endpoint]:
                
                train_set_label = train_test_label_pair[0]
                test_set_label = train_test_label_pair[1]

                train_df_complete,test_df_complete = load_train_test_subsets_from_pkls_without_dropping_cols(dir_with_files_to_parse,train_set_label,test_set_label)

                train_inchis_ids_tuples_list = get_inchis_ids_tuples_list(train_df_complete,inchi_col,id_col)
                test_inchis_ids_tuples_list = get_inchis_ids_tuples_list(test_df_complete,inchi_col,id_col)

                check_inchis_and_ids_are_unique_across_train_test_combination(train_inchis_ids_tuples_list,test_inchis_ids_tuples_list)

                ds_to_ep_to_inchis_ids_tuples_list[ds][endpoint] += train_inchis_ids_tuples_list
                ds_to_ep_to_inchis_ids_tuples_list[ds][endpoint] += test_inchis_ids_tuples_list

    return ds_to_ep_to_inchis_ids_tuples_list


def get_dataset_to_endpoint_to_ids_of_model_ready_datasets(top_class_or_reg_ds_dirs,ds_matched_to_ep_to_all_train_test_pairs,subdir_with_model_ready_datasets='Model_Ready'):
    print(f'Calling this function: {report_name_of_function_where_this_is_called()}')

    ds_ep_to_ids_of_model_ready_datasets = doubleDefaultDictOfLists()

    for ds in top_class_or_reg_ds_dirs.keys():
        top_dir = top_class_or_reg_ds_dirs[ds]

        dir_with_files_to_parse = os.path.join(top_dir,subdir_with_model_ready_datasets)
        
        for endpoint in ds_matched_to_ep_to_all_train_test_pairs[ds]:
            for train_test_label_pair in ds_matched_to_ep_to_all_train_test_pairs[ds][endpoint]:
                
                train_set_label = train_test_label_pair[0]
                test_set_label = train_test_label_pair[1]

                fps_train,  fps_test,  test_ids,  X_train_and_ids_df,  X_test_and_ids_df,  train_y,  test_y,  train_ids = load_ds_from_pkl_files_ready_for_modelling_and_ad_calculations(dir_with_files_to_parse,  train_set_label,  test_set_label)

                ds_ep_to_ids_of_model_ready_datasets[ds][endpoint] += test_ids[:]+train_ids[:]

    return ds_ep_to_ids_of_model_ready_datasets

def filter_ids_dropped_prior_to_modelling(ds_to_ep_to_inchis_ids_tuples_list,ds_ep_to_ids_of_model_ready_datasets):

    ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids = doubleDefaultDictOfLists()

    for ds in ds_to_ep_to_inchis_ids_tuples_list.keys():
        for endpoint in ds_to_ep_to_inchis_ids_tuples_list[ds].keys():
            inchis_ids_tuples_list = ds_to_ep_to_inchis_ids_tuples_list[ds][endpoint]

            ids_of_model_ready_dataset = ds_ep_to_ids_of_model_ready_datasets[ds][endpoint]

            ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids[ds][endpoint] = [t for t in inchis_ids_tuples_list if t[1] in ids_of_model_ready_dataset]

            #-------------------------
            all_ids_prior_to_model_ready_step = [t[1] for t in inchis_ids_tuples_list]
            all_ids_in_model_ready_step_not_present_in_prior_step = [id_ for id_ in ids_of_model_ready_dataset if not id_ in all_ids_prior_to_model_ready_step]
            assert 0 == len(all_ids_in_model_ready_step_not_present_in_prior_step),f'all_ids_in_model_ready_step_not_present_in_prior_step={all_ids_in_model_ready_step_not_present_in_prior_step}'
            #------------------------



    return ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids

def match_ep_pairs_across_datasets_to_overlap_number(ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids,overlap_fraction=False):
    #This should update each endpoint name by appending the dataset name as a prefix before forming endpoint pairs
    #Otherwise, a check for uniqueness of endpoints names across datasets would be required, in principle.
    
    ds_pairwise_overlaps_dict = {}



    for ds_1 in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids.keys():
        for ds_2 in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids.keys():
            for ep_1 in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids[ds_1].keys():
                for ep_2 in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids[ds_2].keys():

                    comparison_key_1 = f'#{ds_1}-{ep_1}'
                    comparison_key_2 = f'{ds_2}-{ep_2}'

                    inchis_1 = [t[0] for t in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids[ds_1][ep_1]]
                    inchis_2 = [t[0] for t in ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids[ds_2][ep_2]]

                    if not overlap_fraction:
                        overlap = len(set(inchis_1).intersection(set(inchis_2)))
                    else:
                        overlap = len(set(inchis_1).intersection(set(inchis_2)))/len(set(inchis_1))#.union(set(inchis_2)))

                    ds_pairwise_overlaps_dict[(comparison_key_1,comparison_key_2)] = overlap

                    

    return ds_pairwise_overlaps_dict

def convert_pairwise_overlaps_dict_to_dataframe(ds_pairwise_overlaps_dict):
    '''
    ds_pairwise_overlaps_dict :
    ds_pairwise_overlaps_dict[(dataset-endpoint combination 1 - string ,dataset-endpoint combination 2 - string)] = <float - the number of compounds in common divided by the number of compounds in the first combination>
    '''
    ######################
    #The concepts used here are discussed in the documentation:
    #https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series
    #https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.from_tuples.html
    #https://pandas.pydata.org/docs/reference/api/pandas.Series.unstack.html
    #####################

    multi_index_series = pd.Series(ds_pairwise_overlaps_dict.values(),index=pd.MultiIndex.from_tuples(ds_pairwise_overlaps_dict.keys()))

    ds_ep_pairwise_overlaps_df = multi_index_series.unstack(level=-1)

    #ds_ep_pairwise_overlaps_df = ds_ep_pairwise_overlaps_df.reset_index()


    return ds_ep_pairwise_overlaps_df

def get_dataset_endpoint_pairwise_overlaps_for_model_ready_datasets(ds_to_ep_to_inchis_ids_tuples_list,ds_ep_to_ids_of_model_ready_datasets,overlap_fraction=True):
    print(f'Calling this function: {report_name_of_function_where_this_is_called()}')

    ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids = filter_ids_dropped_prior_to_modelling(ds_to_ep_to_inchis_ids_tuples_list,ds_ep_to_ids_of_model_ready_datasets)
    
    ds_pairwise_overlaps_dict = match_ep_pairs_across_datasets_to_overlap_number(ds_to_ep_to_inchis_ids_tuples_list_for_only_model_ready_ids,overlap_fraction=overlap_fraction)

    ds_ep_pairwise_overlaps_df = convert_pairwise_overlaps_dict_to_dataframe(ds_pairwise_overlaps_dict)

    return ds_ep_pairwise_overlaps_df

def format_value(v):
    if not isinstance(v,str):
        txt = f'{v:.2f}'
    else:
        txt = v
    
    txt_center_aligned = f'{txt: ^5}'
    
    return txt_center_aligned
    

def style_operations_to_apply_in_oder(styler):

    assert isinstance(styler,pd.io.formats.style.Styler),type(styler)
    
    #Why is precision not allowed vs. documentation? Should we be using a more recent version of Pandas? - "TypeError: format() got an unexpected keyword argument 'precision'""
    styler.format(formatter=format_value,na_rep="Missing") 
    
    styler.background_gradient(axis=None,cmap="hot_r")

    return styler


def style_my_dataframe(ds_ep_pairwise_overlaps_df):
    ###################
    #The following references were consulted to figure out how the style.Styler methods work:
    #https://pandas.pydata.org/docs/user_guide/style.html 
    #https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.pipe.html
    #https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.format.html
    #https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html
    #https://matplotlib.org/stable/users/explain/colors/colormaps.html
    #https://matplotlib.org/stable/gallery/color/colormap_reference.html
    ###################
     
    styled_df = ds_ep_pairwise_overlaps_df

    styler = styled_df.style.pipe(style_operations_to_apply_in_oder)

    #assert isinstance(styled_df,pd.DataFrame),type(styled_df)

    
    return styler

def visualize_dataset_overlaps_of_model_ready_datasets(ds_ep_pairwise_overlaps_df,out_dir):
    ##################
    #https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_excel.html
    ###############################
    
    file_used_to_create_image = os.path.sep.join([out_dir,'dataset_overlaps.xlsx'])

    styler = style_my_dataframe(ds_ep_pairwise_overlaps_df)

    #For now, will need to manually wrap, center and assign 2dp to each cell:
    #Excel View -> Zoom -> 20% will fit whole matrix into one image (if desired)
    styler.to_excel(file_used_to_create_image,freeze_panes=(1,1)) 

def main():
    print('THE START')

    #############
    out_dir = os.path.sep.join([overall_top_public_data_dir,'Dataset_Overlaps'])
    createOrReplaceDir(out_dir)
    #############

    ds_to_ep_to_inchis_ids_tuples_list = get_dataset_to_endpoint_to_inchis_list(top_class_or_reg_ds_dirs,ds_matched_to_ep_to_all_train_test_pairs,subdir_with_pre_model_ready_datasets='Filter3',inchi_col=stand_mol_inchi_col,id_col=consistent_id_col)

    ds_ep_to_ids_of_model_ready_datasets = get_dataset_to_endpoint_to_ids_of_model_ready_datasets(top_class_or_reg_ds_dirs,ds_matched_to_ep_to_all_train_test_pairs,subdir_with_model_ready_datasets='Model_Ready')

    ds_ep_pairwise_overlaps_df = get_dataset_endpoint_pairwise_overlaps_for_model_ready_datasets(ds_to_ep_to_inchis_ids_tuples_list,ds_ep_to_ids_of_model_ready_datasets,overlap_fraction=True)

    visualize_dataset_overlaps_of_model_ready_datasets(ds_ep_pairwise_overlaps_df,out_dir)
    
    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
