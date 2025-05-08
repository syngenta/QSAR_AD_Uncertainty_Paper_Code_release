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
top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','Tox21'])
#--------------------------------------------
smiles_col = 'RDKit_SMILES'
###################################
from common_globals_classification_scripts import all_Tox21_endpoints
#-----------------------------
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.CheminformaticsUtils import chem_data_parsing_utils as ChemDataParser
from modelsADuncertaintyPkg.utils.basic_utils import reportSubDirs

def getAllTox21SdfNames(top_ds_dir):
    all_sdf_names = []
    
    for subdir in reportSubDirs(top_ds_dir):
        print('Looking for SDF files in {}'.format(subdir))
        for sdf_file in glob.glob(os.path.sep.join([subdir,'*.sdf'])):
            all_sdf_names.append(sdf_file)
    
    return all_sdf_names

def getRawSmilesCsvName(sdf_name):
    #--------------------------------------------------
    csv_name = re.sub('(\.sdf$)','_smi.csv',sdf_name)
    assert not csv_name == sdf_name,sdf_name
    #--------------------------------------------------
    return csv_name

def convertAllTox21SDFsToRawSmilesCsvs(top_ds_dir,smiles_col):
    
    all_sdf_names = getAllTox21SdfNames(top_ds_dir)
    
    for sdf_name in all_sdf_names:
    
        df = ChemDataParser.convert_sdf_to_DataFrame_with_SMILES(sdf_name,mol_id='ID',smiles_col_name=smiles_col,drop_rdkit_mol=True)
        
        csv_name = getRawSmilesCsvName(sdf_name)
        
        df.to_csv(csv_name,index=False)
        

def getUpdatedTox21ScoreCsvName(smi_csv):
    #--------------------------------------------------
    csv_name = re.sub('(\.csv$)','_PlusACT.csv',smi_csv)
    assert not csv_name == smi_csv,smi_csv
    #--------------------------------------------------
    return csv_name

def addActivityDataToTox21ScoreCsv(top_ds_dir):
    smi_csv = os.path.sep.join([top_ds_dir,'Tox21Score','tox21_10k_challenge_score_smi.csv'])
    
    act_txt = os.path.sep.join([top_ds_dir,'Tox21Score','tox21_10k_challenge_score.txt'])
    
    smi_df = pd.read_csv(smi_csv)
    
    act_df = pd.read_csv(act_txt,sep="\t")
    
    combined_df = smi_df.merge(act_df,on='Sample ID')
    
    #-----------------------------
    assert combined_df.shape[0]==smi_df.shape[0],f"Rows in merged data={combined_df.shape[0]} vs. rows in SMILES data={smi_df.shape[0]}"
    
    if not combined_df.shape[0]==act_df.shape[0]:
        cmp_str = f"Rows in merged data={combined_df.shape[0]} vs. rows in activity data={act_df.shape[0]}"
        
        assert combined_df.shape[0]<act_df.shape[0],cmp_str
        
        print(cmp_str)
    #-----------------------------
    
    updated_tox21_score_smi_csv = getUpdatedTox21ScoreCsvName(smi_csv)
    
    combined_df.to_csv(updated_tox21_score_smi_csv,index=False)
    
    return updated_tox21_score_smi_csv

def getEndpointSpecificCsvName(smi_csv_to_parse,endpoint):
    #--------------------------------------------------
    csv_name = re.sub('(\.csv$)',f'_{endpoint}.csv',smi_csv_to_parse)
    assert not csv_name == smi_csv_to_parse,smi_csv_to_parse
    #--------------------------------------------------
    return csv_name

def splitDatasetsByEndpoint(smi_csv_to_parse,all_Tox21_endpoints):
    
    all_df = pd.read_csv(smi_csv_to_parse)
    
    for endpoint in all_Tox21_endpoints:
        endpoint_df = all_df[all_df[endpoint].isin(['0','1'])]
        
        endpoint_smi_csv = getEndpointSpecificCsvName(smi_csv_to_parse,endpoint)
        
        endpoint_df.to_csv(endpoint_smi_csv,index=False)

def makeTrainNamesConsistent(top_ds_di,all_Tox21_endpoints):
    for endpoint in all_Tox21_endpoints:
        lower_case_endpoint = endpoint.lower()
        
        orig_csv = os.path.sep.join([top_ds_dir,'Tox21Train',f'{lower_case_endpoint}_smi.csv'])
        
        new_csv = os.path.sep.join([top_ds_dir,'Tox21Train',f'train_ready_{endpoint}.csv'])
        
        df = pd.read_csv(orig_csv)
        
        df_=df.rename({'Active':endpoint},axis=1)
        
        df_.to_csv(new_csv,index=False)

def main():
    print('THE START')
    
    convertAllTox21SDFsToRawSmilesCsvs(top_ds_dir,smiles_col)
    
    updated_tox21_score_smi_csv = addActivityDataToTox21ScoreCsv(top_ds_dir)
    
    for test_set in ['Tox21Score','Tox21Test']:
        if 'Tox21Score' == test_set:
            smi_csv_to_parse = updated_tox21_score_smi_csv
        elif 'Tox21Test' == test_set:
            smi_csv_to_parse = os.path.sep.join([top_ds_dir,test_set,'tox21_10k_challenge_test_smi.csv'])
        else:
            raise Exception(f'Unrecognised test_set={test_set}')
        
        splitDatasetsByEndpoint(smi_csv_to_parse,all_Tox21_endpoints)
    
    makeTrainNamesConsistent(top_ds_dir,all_Tox21_endpoints)
    
    print('THE END')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
