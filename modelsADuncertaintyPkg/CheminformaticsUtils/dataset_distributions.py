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
import pandas as pd
from .similarity import compute_tanimoto_similarity
from .chem_data_parsing_utils import check_fps_df
from ..utils.basic_utils import findDups

def check_ids_list(ids_list):
    assert isinstance(ids_list,list),type(ids_list)
    assert len(ids_list)==len(set(ids_list)),f'Duplicate IDs={findDups(ids_list)}'

def compute_pairwise_distances_for_dataset(fps_df_for_dataset,data_ids_for_dataset,fp_col='fp',distance_metric='Tanimoto'):
    #-----------------
    check_fps_df(fps_df_for_dataset)
    check_ids_list(data_ids_for_dataset)
    assert len(data_ids_for_dataset)==fps_df_for_dataset.shape[0],f'len(data_ids_for_dataset)={len(data_ids_for_dataset)} vs. fps_df_for_dataset.shape[0]={fps_df_for_dataset.shape[0]}'
    #------------------
    
    dataset_pairwise_distances = []
    dict_of_id_pairs_to_pairwise_distances = {}
    
    for j, row_j in fps_df_for_dataset.iterrows():
        
        fp_1 = row_j[0]
        
        for i, row_i in fps_df_for_dataset.iterrows():
            #======================
            if i <= j:
                continue
            #======================
            
            fp_2 = row_i[0]
            
            if 'Tanimoto' == distance_metric:
                sims_dis = 1 - compute_tanimoto_similarity(fp1=fp_1,fp2=fp_2)
            else:
                raise Exception(f'Only the Tanimoto distance has been implemented!')
            
            dataset_pairwise_distances.append(sims_dis)
            
            dict_of_id_pairs_to_pairwise_distances[f'id={data_ids_for_dataset[i]};id={data_ids_for_dataset[j]}'] = sims_dis
    
    return dataset_pairwise_distances,dict_of_id_pairs_to_pairwise_distances

def compute_ds1_vs_ds2_pairwise_distances(fps_df_for_ds1,data_ids_for_ds1,fps_df_for_ds2,data_ids_for_ds2,fp_col='fp'):
    #-----------------
    check_fps_df(fps_df_for_ds1)
    check_fps_df(fps_df_for_ds2)
    check_ids_list(data_ids_for_ds1)
    check_ids_list(data_ids_for_ds2)
    assert len(data_ids_for_ds1)==fps_df_for_ds1.shape[0],f'len(data_ids_for_ds1)={len(data_ids_for_ds1)},vs. fps_df_for_ds1.shape[0]={fps_df_for_ds1.shape[0]}'
    assert len(data_ids_for_ds2)==fps_df_for_ds2.shape[0],f'len(data_ids_for_ds2)={len(data_ids_for_ds2)},vs. fps_df_for_ds2.shape[0]={fps_df_for_ds2.shape[0]}'
    #------------------
    
    ds1_vs_ds2_pairwise_distances = []
    dict_of_id_pairs_to_ds1_vs_ds2_pairwise_distances = {} 
    
    for j, row_j in fps_df_for_ds2.iterrows():
        
        fp_2 = row_j[0] #Curiously,specifying fp_col failed!
        
        for i, row_i in fps_df_for_ds1.iterrows():
            
            fp_1 = row_i[0]
            
            sims_dis = 1 - compute_tanimoto_similarity(fp1=fp_1,fp2=fp_2)
            
            ds1_vs_ds2_pairwise_distances.append(sims_dis)
            
            try: #Debugging
                dict_of_id_pairs_to_ds1_vs_ds2_pairwise_distances[f'id_1={data_ids_for_ds1[i]};id_2={data_ids_for_ds2[j]}'] = sims_dis
            except IndexError:
                raise Exception(f'len(data_ids_for_ds1)={len(data_ids_for_ds1)},len(data_ids_for_ds2)={len(data_ids_for_ds2)}: IndexError occured with i={i},j={j}')
    
    return ds1_vs_ds2_pairwise_distances,dict_of_id_pairs_to_ds1_vs_ds2_pairwise_distances
