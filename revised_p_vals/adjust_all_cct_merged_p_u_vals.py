######################################################
# Syngenta Open Source release: This file has been adapted from code developed in the context of a Syngenta funded collaboration with the University of Sheffield: "Improved Estimation of Prediction Uncertainty Leading to Better Decisions in Crop Protection Research". In some cases, this code is a derivative work of other Open Source code. Please see under "If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below" for further details.
#Copyright (c) 2021-2026  Syngenta
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
###########################################################
import os,re,glob,sys
import pandas as pd
#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(dir_of_this_file)
#==========================================
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import report_subdirs_recursively
from modelsADuncertaintyPkg.qsar_eval.aggregate_p_vals import aggregate_p_vals
#----------------------------------------------
from scripts.adjust_all_p_values import get_all_raw_p_vals_df_dict,adjust_all_p_values,write_all_adjusted_p_vals_csvs
#-----------------------------------------------
overall_top_public_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])
overall_top_syn_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData'])
#------------------------------------------------
#=================================================
#Must be consistent with above:
pre_merge_p_vals_col='all_p_vals_str'
raw_p_vals_col='Shift-Metric P-value'
adj_p_vals_col='GLOBAL Adjusted P-value'
#====================================================

def check_p_u(p_u):
    #Based on https://arxiv.org/abs/1603.05766v1

    if not pd.isna(p_u):
        assert isinstance(p_u,float)
        assert p_u <= 1 and p_u > 0,p_u

def check_a_list_of_p_u_vals(list_of_p_u_vals):
    assert isinstance(list_of_p_u_vals,list)

    for p_u in list_of_p_u_vals:
        check_p_u(p_u)

def replace_default_merged_p_val_with_cct(row,pre_merge_p_vals_col,raw_p_vals_col):

    pre_merge_p_vals = [float(v) for v in row[pre_merge_p_vals_col].split(';')]

    check_a_list_of_p_u_vals(list_of_p_u_vals=pre_merge_p_vals)

    new_merged_p_val = aggregate_p_vals(list_of_p_vals=pre_merge_p_vals,p_val_aggregation_option="cct")

    row[raw_p_vals_col] = new_merged_p_val

    return row


def update_p_u_files_with_new_merged_p_vals_if_applicable(old_p_val_file,approach='cct_merge_p_u_vals'):

    new_p_vals_file = re.sub('(\.csv$)',f'_{approach}.csv',old_p_val_file)
    assert not new_p_vals_file == old_p_val_file

    df = pd.read_csv(old_p_val_file)

    if 'cct_merge_p_u_vals' == approach:
        df = df.apply(replace_default_merged_p_val_with_cct,axis=1,args=(pre_merge_p_vals_col,raw_p_vals_col))
    else:
        raise Exception(f'Unrecognised approach={approach}')

    df.to_csv(new_p_vals_file,index=False)



def update_p_u_files_with_new_merged_p_vals_if_applicable_in_all_old_p_vals_files(overall_top_public_data_dir,overall_top_syn_data_dir,approach='cct_merge_p_u_vals',p_vals_csv_suffix='_p_u_vals.csv'):

    for top_dataset_dir in [overall_top_public_data_dir,overall_top_syn_data_dir]:
        for subdir in report_subdirs_recursively(dir_=top_dataset_dir,absolute_names=True):
                        
            subdir_specific_relevant_p_vals_files = glob.glob(os.path.sep.join([subdir,f'*{p_vals_csv_suffix}']))

            for old_p_val_file in subdir_specific_relevant_p_vals_files:

                #================================
                if approach in os.path.basename(old_p_val_file):
                    print(f'Ignoring this file from a previous run of this script before updating the merged p_u values: {old_p_val_file}')
                    continue
                #================================

                update_p_u_files_with_new_merged_p_vals_if_applicable(old_p_val_file,approach=approach)

def get_minimum_p(all_p_vals_df_dict,p_vals_col):

    all_p_vals = []

    for old_raw_p_vals_file_name in all_p_vals_df_dict.keys():
        print(f'Searching for {p_vals_col} p-values in p-values file corresponding to {old_raw_p_vals_file_name}')
        p_vals = all_p_vals_df_dict[old_raw_p_vals_file_name][p_vals_col].tolist()

        all_p_vals += p_vals
    
    minimum_p = min(all_p_vals)

    return minimum_p

def report_minimum_p_val(one_tail_family,all_p_vals_df_dict,approach,p_vals_col):

    minimum_p = get_minimum_p(all_p_vals_df_dict,p_vals_col)

    if p_vals_col == adj_p_vals_col:
        type_of_p = 'adjusted'
    elif p_vals_col == raw_p_vals_col:
        type_of_p = 'raw'
    else:
        raise Exception(f'p_vals_col={p_vals_col}')
    
    print(f'approach={approach}: minimum {type_of_p} p-value (out of one-tail p-values = {one_tail_family}) = {minimum_p}')


def check_that_all_raw_p_vals_df_dict_does_not_include_extra_files(one_tail_family,all_raw_p_vals_df_dict,approach):
    for file_name in all_raw_p_vals_df_dict.keys():
        assert approach in os.path.basename(file_name),file_name
        assert not f'_{approach}_{approach}' in file_name,file_name
        print('='*50)
        print(f'The p-values in the following file will be treated as part of the family (one-tail={one_tail_family}) for which multiple testing adjustement will be performed:')
        print(file_name)
        print('='*50)

def filter_metrics(all_raw_p_vals_df_dict,metrics_to_filter=['Efficiency']):

    filtered_all_raw_p_vals_df_dict = {}

    for p_vals_file in all_raw_p_vals_df_dict.keys():
        df = all_raw_p_vals_df_dict[p_vals_file]
        count = df.shape[0]
        df = df[~df['Metric'].isin(metrics_to_filter)].reset_index(inplace=False,drop=True)
        new_count = df.shape[0]
        if 'regression' in os.path.basename(p_vals_file).lower():
            assert new_count < count,p_vals_file

        filtered_all_raw_p_vals_df_dict[p_vals_file] = df

    
    return filtered_all_raw_p_vals_df_dict

def main():
    print('THE START')

    for approach in ['cct_merge_p_u_vals']:
        update_p_u_files_with_new_merged_p_vals_if_applicable_in_all_old_p_vals_files(overall_top_public_data_dir,overall_top_syn_data_dir,approach=approach)

        for one_tail_family in [True,False]:

            all_raw_p_vals_df_dict = get_all_raw_p_vals_df_dict(one_tail_family,all_top_dirs_containing_raw_p_vals_csvs=[overall_top_public_data_dir,overall_top_syn_data_dir],p_vals_csv_suffix=f'_{approach}.csv')

            check_that_all_raw_p_vals_df_dict_does_not_include_extra_files(one_tail_family,all_raw_p_vals_df_dict,approach)

            all_raw_p_vals_df_dict = filter_metrics(all_raw_p_vals_df_dict)
            
            if 'cct_merge_p_u_vals' == approach:
                all_adjusted_p_vals_df_dict = adjust_all_p_values(all_raw_p_vals_df_dict)
            else:
                raise Exception(f'Unrecognised approach = {approach}')


            write_all_adjusted_p_vals_csvs(all_adjusted_p_vals_df_dict,adjusted_p_vals_csv_suffix='_new_GlobalAdjusted.csv')

            report_minimum_p_val(one_tail_family,all_adjusted_p_vals_df_dict,approach,adj_p_vals_col)

            report_minimum_p_val(one_tail_family,all_raw_p_vals_df_dict,approach,raw_p_vals_col)

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
