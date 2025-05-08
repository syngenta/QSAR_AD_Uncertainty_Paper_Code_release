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
import os,sys,glob,re
import pandas as pd
from common_globals import consistent_id_col,consistent_act_col,consistent_smiles_col,consistent_inchi_col,top_dir_of_public_data_scripts,pkg_dir,top_class_or_reg_ds_dirs,regression_dataset_names,classification_dataset_names,ds_matched_to_ep_list,get_endpoint_type
from common_globals import all_Wang_endpoints,wang_raw_smiles_col,wang_raw_act_col,wang_ids_col,wang_test_set_names,no_wang_outer_folds
from common_globals import all_Tox21_endpoints,all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.utils.basic_utils import check_no_missing_values
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir
#-------------------------------------------


def should_we_filter_this_smiles(row,consistent_smiles_col,consistent_id_col,filter_col,heavy_atoms_limit=4):
    ##########################
    #This is based upon, but not identical to, "Data preprocessing" in Morger et al. (2021) [https://link.springer.com/article/10.1186/s13321-021-00511-5]
    #Standardization, which may be performed somewhat differently, and identification and removal of InChI duplicates before and after standardization are handled elsewhere.
    ##########################
    
    row[filter_col] = False
    
    try:
        mol = ChemDataParser.Chem.MolFromSmiles(row[consistent_smiles_col])
    
        if ChemDataParser.this_is_a_non_organic_compound(mol):
            row[filter_col] = True
        
        components = ChemDataParser.get_all_non_salt_mol_component_as_molecules(mol)
        
        if len(components) > 1:
            row[filter_col] = True
        else:
            remaining_component = components[0]
            
            if remaining_component.GetNumHeavyAtoms() < heavy_atoms_limit:
                row[filter_col] = True
    except Exception as err:
        print(f'WARNING: Molecule with ID = {row[consistent_id_col]} could not be parsed (error = {err}), so will automatically be filtered!')
        
        row[filter_col] = True
    
    return row

def identify_smiles_to_filter(df,consistent_smiles_col,consistent_id_col,filter_col):
    
    df=df.apply(should_we_filter_this_smiles,axis=1,args=(consistent_smiles_col,consistent_id_col,filter_col))
    
    return df

def filter_smiles_based_upon_new_rules(csv_file,consistent_smiles_col,consistent_id_col,dir_of_new_filtered_files,filter_col='New_Rules_Excluded'):
    new_file = os.path.sep.join([dir_of_new_filtered_files,os.path.basename(re.sub('(\.csv$)','_NRsFilt.csv',csv_file))])
    assert not new_file == csv_file,csv_file
    
    df = pd.read_csv(csv_file)
    
    input_no_rows = df.shape[0]
    
    df = identify_smiles_to_filter(df,consistent_smiles_col,consistent_id_col,filter_col)
    
    assert df.shape[0] == input_no_rows,csv_file
    
    df=df[~df[filter_col].isin([True])]
    
    df = df.drop(labels=[filter_col],axis=1)
    
    assert df.shape[0] <= input_no_rows,csv_file
    
    if not df.shape[0] == input_no_rows:
        print(f'New rules excluded {df.shape[0]-input_no_rows} compounds from {csv_file}')
    
    df.to_csv(new_file,index=False)

def filter_all_files_based_upon_new_rules(dir_with_files_to_filter,consistent_smiles_col,consistent_id_col,dir_of_new_filtered_files):
    
    for csv_file in glob.glob(os.path.sep.join([dir_with_files_to_filter,'*.csv'])):
        filter_smiles_based_upon_new_rules(csv_file,consistent_smiles_col,consistent_id_col,dir_of_new_filtered_files)

def main():
    print('THE START')
    
    for dataset_name in top_class_or_reg_ds_dirs.keys():
        
        endpoint_type = get_endpoint_type(dataset_name,regression_dataset_names,classification_dataset_names)
        
        dir_with_files_to_filter = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter1'])
        
        dir_of_new_filtered_files = os.path.sep.join([top_class_or_reg_ds_dirs[dataset_name],'Filter2'])
        
        createOrReplaceDir(dir_=dir_of_new_filtered_files)
        
        filter_all_files_based_upon_new_rules(dir_with_files_to_filter,consistent_smiles_col,consistent_id_col,dir_of_new_filtered_files)
    
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

