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
########################################################
#Copyright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#########################################################
import os,re,glob,sys
import pandas as pd
#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(dir_of_this_file)
#==========================================
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.qsar_eval.adjust_p_vals import getAdjustedPValues
from modelsADuncertaintyPkg.utils.basic_utils import report_subdirs_recursively
#-----------------------------------------------
overall_top_public_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData'])
overall_top_syn_data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'SyngentaData'])
#------------------------------------------------

def get_all_raw_p_vals_df_dict(all_top_dirs_containing_raw_p_vals_csvs=[overall_top_public_data_dir,overall_top_syn_data_dir],p_vals_csv_suffix='_PVals.csv'):

    all_raw_p_vals_df_dict = {}
    for top_dataset_dir in all_top_dirs_containing_raw_p_vals_csvs:
        for subdir in report_subdirs_recursively(dir_=top_dataset_dir,absolute_names=True):
            #print(f'Looking for p-value files inside subdir={subdir}')
            for p_val_file in glob.glob(os.path.sep.join([subdir,f'*{p_vals_csv_suffix}'])):
                print(f'Considering p_val_file={p_val_file}')
                all_raw_p_vals_df_dict[p_val_file] = pd.read_csv(p_val_file)


    return all_raw_p_vals_df_dict

def adjust_all_p_values(all_raw_p_vals_df_dict,raw_p_vals_col='Shift-Metric P-value',adj_p_vals_col='GLOBAL Adjusted P-value',sig_level_perc=5,p_val_adjustment_method='fdr_by',skip_missing=True,debug=False):
    all_adjusted_p_vals_df_dict = {}

    

    all_raw_p_vals = []

    for file_name in all_raw_p_vals_df_dict.keys():
        

        df = all_raw_p_vals_df_dict[file_name]

        all_raw_p_vals += df[raw_p_vals_col].tolist()


    all_adjusted_p_vals_list = getAdjustedPValues(p_vals_list=all_raw_p_vals,sig_level_perc=sig_level_perc,p_val_adjustment_method=p_val_adjustment_method,skip_missing=skip_missing)

    start = 0
    for file_name in all_raw_p_vals_df_dict.keys():

        

        df = all_raw_p_vals_df_dict[file_name]

        size = df.shape[0]

        relevant_adjusted_p_vals_list = all_adjusted_p_vals_list[start:(start+size)]

        assert len(relevant_adjusted_p_vals_list) == size,f'file_name={file_name},len(relevant_adjusted_p_vals_list)={len(relevant_adjusted_p_vals_list)},size={size}'

        #-----------------------
        if debug:
            print(f'File name={file_name}')
            print(f'relevant_adjusted_p_vals_list={relevant_adjusted_p_vals_list}')
        #-----------------------
        
        
        df.insert(loc=df.shape[1],column=adj_p_vals_col,value=pd.Series(relevant_adjusted_p_vals_list),allow_duplicates=False)

        all_adjusted_p_vals_df_dict[file_name] = df

        start += size

    return all_adjusted_p_vals_df_dict

def write_all_adjusted_p_vals_csvs(all_adjusted_p_vals_df_dict,adjusted_p_vals_csv_suffix='_GlobalAdjusted.csv'):
    for old_file_name in all_adjusted_p_vals_df_dict.keys():
        new_file_name = re.sub('(\.csv$)',adjusted_p_vals_csv_suffix,old_file_name)
        assert not old_file_name == new_file_name,old_file_name

        df = all_adjusted_p_vals_df_dict[old_file_name]

        df.to_csv(new_file_name,index=False)

def main():
    print('THE START')

    all_raw_p_vals_df_dict = get_all_raw_p_vals_df_dict(all_top_dirs_containing_raw_p_vals_csvs=[overall_top_public_data_dir,overall_top_syn_data_dir],p_vals_csv_suffix='_PVals.csv')

    all_adjusted_p_vals_df_dict = adjust_all_p_values(all_raw_p_vals_df_dict)

    write_all_adjusted_p_vals_csvs(all_adjusted_p_vals_df_dict,adjusted_p_vals_csv_suffix='_GlobalAdjusted.csv')


    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())


