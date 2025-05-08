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
import os,sys,glob,re#,time,pickle,shutil
from collections import defaultdict
import pandas as pd
import numpy as np
##################################
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','Morger_ChEMBL'])
#--------------------------------------------
from common_globals_classification_scripts import all_ChEMBL_endpoints,chembl_ids_col,chembl_smiles_col,chembl_act_class_col,chembl_subsets_of_interest
from loadMorgerChEMBLDataPlusTemporalSplits import prepare_ChEMBL_train_plus_temporal_splits
#-----------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import findDups

def checkAllIdsUnique(endpoint,subset2df):
    all_ids = []
    
    for subset in subset2df.keys():
        df = subset2df[subset]
        
        all_ids += df[chembl_ids_col].tolist()
    
    assert len(all_ids)==len(set(all_ids)),f"endpoint={endpoint}. There are duplicate IDs={findDups(all_ids)}"

def checkTemporalSplit(endpoint,subset2df,chembl_subsets_of_interest):
    
    earliest_to_latest_subset = chembl_subsets_of_interest
    
    for sub_index in range(0,len(earliest_to_latest_subset)):
        if not 0 == sub_index:
            previous_sub_index = sub_index -1
            
            years = subset2df[earliest_to_latest_subset[sub_index]]['year'].unique().tolist()
            
            previous_years = subset2df[earliest_to_latest_subset[previous_sub_index]]['year'].unique().tolist()
            
            assert all([isinstance(a_year,float) for a_year in previous_years+years]),f"endpoint={endpoint},subset={earliest_to_latest_subset[sub_index]},prevous subset={earliest_to_latest_subset[previous_sub_index]},types of year values={[type(a_year) for a_year in previous_years+years]}"
            
            for a_year in years:
                assert all([(a_year>old_year) for old_year in previous_years]),f"endpoint={endpoint},subset={earliest_to_latest_subset[sub_index]},prevous subset={earliest_to_latest_subset[previous_sub_index]}, current year = {a_year}, previous_years={previous_years}"

def main():
    print('THE START')
    
    data_chembl_path = top_ds_dir
    
    for endpoint in all_ChEMBL_endpoints:
        print(f'endpoint={endpoint}')
        
        subset2df = prepare_ChEMBL_train_plus_temporal_splits(endpoint,data_chembl_path)
        
        checkAllIdsUnique(endpoint,subset2df)
        
        checkTemporalSplit(endpoint,subset2df,chembl_subsets_of_interest)
        
        for subset in chembl_subsets_of_interest:
            df = subset2df[subset]
            
            df.to_csv(os.path.sep.join([data_chembl_path,f'{endpoint}_{subset}.csv']),index=False)
            
            
    
    print('THE END')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
