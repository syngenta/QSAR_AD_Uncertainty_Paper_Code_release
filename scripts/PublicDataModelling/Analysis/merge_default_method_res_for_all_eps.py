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
import os,sys,re
import pandas as pd
import numpy as np
from collections import defaultdict

from AD_ranking import filter_to_only_keep_default_AD_params_results

dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_code_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_script)))
sys.path.append(top_code_dir)
from scripts.recommended_defaults import ep_type_matched_to_default_AD_uncertainty_methods
from scripts.consistent_parameters_for_all_modelling_runs import ad_params_col
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir

top_res_dir = os.path.dirname(os.path.dirname(top_code_dir))

dir_with_other_eps_metrics_csvs = os.path.sep.join([top_res_dir,'PublicData','OtherMergedStats'])
dir_with_other_eps_p_vals_csvs = os.path.sep.join([top_res_dir,'PublicData','Other_EPs_AD_P_vals'])
dir_with_exemplar_eps_metrics_csvs = os.path.sep.join([top_res_dir,'PublicData','ModellingResSummary'])

modelling_type_to_dataset_group_names = {}
modelling_type_to_dataset_group_names['Classification'] = ['Morger_ChEMBL','Tox21']
modelling_type_to_dataset_group_names['Regression'] = ['Wang_ChEMBL']

dataset_group_to_dir_with_exemplar_eps_p_vals_csvs = {}
for dataset_group_name in ['Morger_ChEMBL','Tox21','Wang_ChEMBL']:
    if not 'Wang_ChEMBL' == dataset_group_name:
        dataset_group_to_dir_with_exemplar_eps_p_vals_csvs[dataset_group_name] = os.path.sep.join([top_res_dir,'PublicData',dataset_group_name,'Modelling'])
    else:
        dataset_group_to_dir_with_exemplar_eps_p_vals_csvs[dataset_group_name] = os.path.sep.join([top_res_dir,'PublicData',dataset_group_name,'data','dataset','Modelling'])

dir_with_all_eps_metrics_csvs = os.path.sep.join([top_res_dir,'PublicData','AllMergedStats'])
dir_with_all_eps_p_vals_csvs = os.path.sep.join([top_res_dir,'PublicData','All_EPs_AD_P_vals'])

def load_other_eps_metrics(modelling_type,dir_with_other_eps_metrics_csvs):

    other_eps_metrics_csv = os.path.sep.join([dir_with_other_eps_metrics_csvs,f'Other_Endpoints_{modelling_type}_Stats.csv'])

    other_eps_metrics_df = pd.read_csv(other_eps_metrics_csv)

    return other_eps_metrics_df

def load_exemplar_eps_metrics(modelling_type,dir_with_exemplar_eps_metrics_csvs):

    exemplar_eps_metrics_csv = os.path.sep.join([dir_with_exemplar_eps_metrics_csvs,f'Exemplar_Endpoints_{modelling_type}_Stats.csv'])

    exemplar_eps_metrics_including_other_ad_methods_and_params_df = pd.read_csv(exemplar_eps_metrics_csv)

    return exemplar_eps_metrics_including_other_ad_methods_and_params_df

def filter_to_only_keep_default_AD__and_uncertainty_method_metrics(exemplar_eps_metrics_for_default_ad_param_df,modelling_type,ad_method_col='AD Method',uncertainty_method_col='Modelling Algorithm'):

    default_ad_method = ep_type_matched_to_default_AD_uncertainty_methods[modelling_type]['AD_method_name']
    default_uncertainty_method = ep_type_matched_to_default_AD_uncertainty_methods[modelling_type]['uncertainty_method']

    #Also, keep the 'All' subset of results generated prior to AD splitting of the test set
    exemplar_eps_metrics_for_default_ad_method_and_param_df = exemplar_eps_metrics_for_default_ad_param_df[exemplar_eps_metrics_for_default_ad_param_df[ad_method_col].isin([default_ad_method,np.nan])]
    exemplar_eps_metrics_for_AD__and_uncertainty_method_df = exemplar_eps_metrics_for_default_ad_method_and_param_df[exemplar_eps_metrics_for_default_ad_method_and_param_df[uncertainty_method_col].isin([default_uncertainty_method])]


    return exemplar_eps_metrics_for_AD__and_uncertainty_method_df

def filter_metrics_to_only_keep_default_methods_and_params(exemplar_eps_metrics_including_other_ad_methods_and_params_df,modelling_type):

    #although we initially generated results for different AD parameters, we only evaluated AD methods according to results obtained with default parameters, primarily chosen based on considering the trends in percentages lying inside the domain for random test sets
    exemplar_eps_metrics_for_default_ad_param_df = filter_to_only_keep_default_AD_params_results(raw_df=exemplar_eps_metrics_including_other_ad_methods_and_params_df)

    #we also briefly compared different AD methods before concluding there was no overall winner and nUNC was a reasonable default
    exemplar_eps_metrics_for_AD__and_uncertainty_method_df = filter_to_only_keep_default_AD__and_uncertainty_method_metrics(exemplar_eps_metrics_for_default_ad_param_df,modelling_type)

    return exemplar_eps_metrics_for_AD__and_uncertainty_method_df

def load_other_eps_p_vals(modelling_type,p_val_prefix,dir_with_other_eps_p_vals_csvs):

    other_eps_p_vals_csv = os.path.sep.join([dir_with_other_eps_p_vals_csvs,f'{p_val_prefix}{modelling_type}_PVals_GlobalAdjusted.csv'])

    other_eps_p_vals_df = pd.read_csv(other_eps_p_vals_csv)

    return other_eps_p_vals_df

def load_exemplar_eps_p_vals(modelling_type,p_val_prefix,dir_with_exemplar_eps_p_vals):

    if 'Classification' == modelling_type:
        bespoke_name_for_modelling_type = 'binary_class'
    elif 'Regression' == modelling_type:
        bespoke_name_for_modelling_type = 'regression'
    else:
        raise Exception(f'Unrecognised modelling_type={modelling_type}')

    exemplar_eps_p_vals_including_other_ad_methods_csv = os.path.sep.join([dir_with_exemplar_eps_p_vals,f'{p_val_prefix}{bespoke_name_for_modelling_type}_PVals_GlobalAdjusted.csv'])
    
    exemplar_eps_p_vals_including_other_ad_methods_df = pd.read_csv(exemplar_eps_p_vals_including_other_ad_methods_csv)

    return exemplar_eps_p_vals_including_other_ad_methods_df

def this_row_corresponds_to_default_methods(row,default_ad_method,default_uncertainty_method):

    scenario = row['Scenario']
 
    ep,test_set,generic_test_label,generic_FPs_label,ad_method,uncertainty_method =  scenario.split('_')

    return (ad_method == default_ad_method and uncertainty_method == default_uncertainty_method)
        
def filter_p_vals_to_only_keep_default_methods(exemplar_eps_p_vals_including_other_ad_methods_df,modelling_type):

    default_ad_method = ep_type_matched_to_default_AD_uncertainty_methods[modelling_type]['AD_method_name']
    default_uncertainty_method = ep_type_matched_to_default_AD_uncertainty_methods[modelling_type]['uncertainty_method']

    exemplar_eps_p_vals_only_default_methods_df = exemplar_eps_p_vals_including_other_ad_methods_df[exemplar_eps_p_vals_including_other_ad_methods_df.apply(this_row_corresponds_to_default_methods,axis=1,args=(default_ad_method,default_uncertainty_method))]


    return exemplar_eps_p_vals_only_default_methods_df

def main():
    
    print(f'Creating or replacing: {dir_with_all_eps_metrics_csvs}')
    print(f'Creating or replacing: {dir_with_all_eps_p_vals_csvs}')
    

    createOrReplaceDir(dir_with_all_eps_metrics_csvs)
    createOrReplaceDir(dir_with_all_eps_p_vals_csvs)

    
    for modelling_type in modelling_type_to_dataset_group_names.keys():
        all_eps_metrics_dfs_to_merge = []

        

        other_eps_metrics_df = load_other_eps_metrics(modelling_type,dir_with_other_eps_metrics_csvs)

        all_eps_metrics_dfs_to_merge.append(other_eps_metrics_df)

        exemplar_eps_metrics_including_other_ad_methods_and_params_df = load_exemplar_eps_metrics(modelling_type,dir_with_exemplar_eps_metrics_csvs)

        exemplar_eps_metrics_for_AD__and_uncertainty_method_df = filter_metrics_to_only_keep_default_methods_and_params(exemplar_eps_metrics_including_other_ad_methods_and_params_df,modelling_type)

        all_eps_metrics_dfs_to_merge.append(exemplar_eps_metrics_for_AD__and_uncertainty_method_df)

        all_eps_metrics_merged_df = pd.concat(all_eps_metrics_dfs_to_merge,axis=0).reset_index(inplace=False,drop=True)

        all_eps_metrics_merged_csv = os.path.sep.join([dir_with_all_eps_metrics_csvs,f'All_Endpoints_{modelling_type}_Stats.csv'])

        #ad_params_col for exemplar datasets subset was already dropped during filtering to only keep the default parameters
        assert all([v for v in all_eps_metrics_merged_df[ad_params_col].tolist() if v in ['k=3_t=0.4','']])
        
        #tidy up the final merged metrics CSV file
        #all_eps_metrics_merged_df = all_eps_metrics_merged_df[~all_eps_metrics_merged_df['AD Subset'].isin(['All'])]

        all_eps_metrics_merged_df = all_eps_metrics_merged_df.drop(ad_params_col,axis=1)

        all_eps_metrics_merged_df.to_csv(all_eps_metrics_merged_csv,index=False)

        for p_val_type in ['one_tail','two_tail']:
            all_eps_p_vals_df_to_merge = []

            if 'one_tail' == p_val_type:
                p_val_prefix = 'one_tail_'
            elif 'two_tail' == p_val_type:
                p_val_prefix = ''
            else:
                raise Exception(f'Unrecognised p_val_type={p_val_type}') 

            other_eps_p_vals_df = load_other_eps_p_vals(modelling_type,p_val_prefix,dir_with_other_eps_p_vals_csvs)

            all_eps_p_vals_df_to_merge.append(other_eps_p_vals_df)

            
            
            
            for dataset_group_name in modelling_type_to_dataset_group_names[modelling_type]:

                dir_with_exemplar_eps_p_vals = dataset_group_to_dir_with_exemplar_eps_p_vals_csvs[dataset_group_name]

                exemplar_eps_p_vals_including_other_ad_methods_df = load_exemplar_eps_p_vals(modelling_type,p_val_prefix,dir_with_exemplar_eps_p_vals)

                exemplar_eps_p_vals_only_default_methods_df = filter_p_vals_to_only_keep_default_methods(exemplar_eps_p_vals_including_other_ad_methods_df,modelling_type)

                all_eps_p_vals_df_to_merge.append(exemplar_eps_p_vals_only_default_methods_df)
            
            

            all_eps_p_vals_merged_df = pd.concat(all_eps_p_vals_df_to_merge,axis=0).reset_index(inplace=False,drop=True)

            all_eps_p_vals_merged_csv = os.path.sep.join([dir_with_all_eps_p_vals_csvs,f'{p_val_prefix}{modelling_type}_PVals_GlobalAdjusted.csv'])


            all_eps_p_vals_merged_df.to_csv(all_eps_p_vals_merged_csv,index=False)






    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
