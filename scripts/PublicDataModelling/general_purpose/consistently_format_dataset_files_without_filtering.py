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
#############################################################
#To prepare this script, filter_duplicates_plus_retain_InChIs_for_all_datasets.py was copied and adapted to ensure consistent organization of output files, in a different folder, with just the filtering steps removed
#############################################################
import os,sys
import pandas as pd
#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
top_dir_of_public_data_scripts = os.path.dirname(dir_of_this_file)
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_file)))
#-----------------------------------------------
overall_top_public_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])
top_class_or_reg_ds_dirs = {}
top_class_or_reg_ds_dirs['Morger_Tox21'] = os.path.sep.join([overall_top_public_data_dir,'Tox21'])
top_class_or_reg_ds_dirs['Morger_ChEMBL'] = os.path.sep.join([overall_top_public_data_dir,'Morger_ChEMBL'])
top_class_or_reg_ds_dirs['Wang_ChEMBL'] = os.path.sep.join([overall_top_public_data_dir,'Wang_ChEMBL','data','dataset'])
regression_dataset_names = ['Wang_ChEMBL']
classification_dataset_names = ['Morger_ChEMBL','Morger_Tox21']
#-----------------------------------------------
sys.path.append(pkg_dir)
#from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.utils.basic_utils import check_no_missing_values
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
#-------------------------------------------
sys.path.append(top_dir_of_public_data_scripts)
from regression.common_globals_regression_scripts import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_set_names,no_wang_outer_folds
from classification.common_globals_classification_scripts import all_Tox21_endpoints,all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest,tox21_ids_col,tox21_smiles_col
from common_globals import consistent_id_col,consistent_act_col,consistent_smiles_col,consistent_inchi_col,ds_matched_to_ep_list,get_endpoint_type,subset_name_col
#-------------------------------------------
#Import functions from filter_duplicates_plus_retain_InChIs_for_all_datasets.py so any fixes are applied consistently:
sys.path.append(dir_of_this_file)
from filter_duplicates_plus_retain_InChIs_for_all_datasets import add_subset_name,make_column_names_consistent,update_list_of_subsets_df_ready_to_concatenate,combine_all_subsets_annotated_with_subset_name,write_filtered_subset_files
#--------------------------------------------

def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        endpoint_type = get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)
        
        out_dir = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'NoFilt'])
        
        createOrReplaceDir(dir_=out_dir)
        
        for endpoint in ds_matched_to_ep_list[dataset_name]:
            #########################################
            #For the Wang dataset, for a given endpoint, the [train+test] molecules for a given [test_name (= ivit or ivot) & fold] combination correspond to the entire dataset, with another [test_name (= ivit or ivot) & fold] combination just corresponding to a different distribution of the molecules between the train and test sets. Hence, we need to combine all molecules and remove duplicates separately for each [test_name (= ivit or ivot) & fold] combination.
            #########################################
            for wang_test_name in wang_test_set_names:
                for wang_fold in range(0,no_wang_outer_folds):
                    #------------------------------------------
                    if not 'Wang_ChEMBL' == dataset_name:
                        if not (wang_test_name == wang_test_set_names[0] and wang_fold == 0):
                            continue
                    #-------------------------------------------
                    
                    ###################################################
                    #Making the names of all columns consistent across the raw file results and removing other columns will be necessary prior to assigning a subset_name_col and then concatenating the dataframes:
                    combined_df = combine_all_subsets_annotated_with_subset_name(top_class_or_reg_ds_dirs,dataset_name,endpoint,subset_name_col,consistent_id_col,consistent_act_col,consistent_smiles_col,tox21_ids_col,tox21_smiles_col,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_name,wang_fold)
                    ###################################################
                    
                    combined_with_final_cols_df = combined_df[[subset_name_col,consistent_id_col,consistent_act_col,consistent_smiles_col]]
                    
                    check_no_missing_values(combined_with_final_cols_df)
                    
                    write_filtered_subset_files(endpoint,subset_name_col,combined_with_final_cols_df,out_dir)
            
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())




