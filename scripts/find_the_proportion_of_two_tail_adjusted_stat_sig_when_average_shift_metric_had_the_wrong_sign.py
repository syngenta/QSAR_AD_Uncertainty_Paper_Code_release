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
from collections import defaultdict
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

def get_all_one_tail_global_adjusted_p_vals_files(all_top_dirs_containing_raw_p_vals_csvs=[overall_top_public_data_dir,overall_top_syn_data_dir]):

    all_one_tail_global_adjusted_p_vals_files = []

    for top_dataset_dir in all_top_dirs_containing_raw_p_vals_csvs:
        for subdir in report_subdirs_recursively(dir_=top_dataset_dir,absolute_names=True):
            #print(f'Looking for p-value files inside subdir={subdir}')
            for p_val_file in glob.glob(os.path.sep.join([subdir,f'one_tail_*_PVals_GlobalAdjusted.csv'])):
                print(f'Considering p_val_file={p_val_file}')
                all_one_tail_global_adjusted_p_vals_files.append(p_val_file)


    return all_one_tail_global_adjusted_p_vals_files


def update_counts(row,counts_dict,sig_level=0.05):
    if pd.isna(row['GLOBAL Adjusted P-value']):
        assert pd.isna(row['one-tail-Shift-Metric P-value']),f"Scenario={row['Scenario']},Metric={row['Metric']}"
        assert pd.isna(row['one-tail-GLOBAL Adjusted P-value']),f"Scenario={row['Scenario']},Metric={row['Metric']}"
        assert pd.isna(row['Shift-Metric P-value']),f"Scenario={row['Scenario']},Metric={row['Metric']}"
        pass
    else:
        counts_dict['total_no_scenarios_for_which_p_vals_were_computed'] += 1

        if round(row['GLOBAL Adjusted P-value'],2) <= sig_level or round(row['one-tail-GLOBAL Adjusted P-value'],2) <= sig_level:
            counts_dict['total_no_with_adjusted_stat_sig'] += 1

        if round(row['GLOBAL Adjusted P-value'],2) <= sig_level and row['average_shift_metric_val_has_wrong_sign']:
            counts_dict['total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign'] += 1
            print(f"Found a scenario, metric combination of interest : Scenario={row['Scenario']},Metric={row['Metric']}")

        #Curiosity:
        if round(row['one-tail-GLOBAL Adjusted P-value'],2) <= sig_level and not row['average_shift_metric_val_has_wrong_sign']:
            counts_dict['total_no_with_one_tail_adjusted_stat_sig_when_average_shift_metric_did_not_have_the_wrong_sign'] += 1


def count_p_vals_of_interest(all_one_tail_global_adjusted_p_vals_files):

    counts_dict = defaultdict(int)

    

    for one_tail_file in all_one_tail_global_adjusted_p_vals_files:
        one_tail_df = pd.read_csv(one_tail_file)

        one_tail_df = one_tail_df.rename(mapper={'GLOBAL Adjusted P-value':'one-tail-GLOBAL Adjusted P-value','Shift-Metric P-value':'one-tail-Shift-Metric P-value'},axis=1)

        two_tail_file = one_tail_file.replace('one_tail_','')

        assert not two_tail_file == one_tail_file,one_tail_file

        two_tail_df = pd.read_csv(two_tail_file)

        #average_shift_metric_val_has_wrong_sign is not relevant to two-tail p-values and, hence, is not populated in these files:
        two_tail_df = two_tail_df.drop(labels='average_shift_metric_val_has_wrong_sign',axis=1)

        merged_df = one_tail_df.merge(two_tail_df,on=['Scenario','Metric'])

        merged_df.apply(update_counts,axis=1,args=(counts_dict,))

    total_no_scenarios_for_which_p_vals_were_computed = counts_dict['total_no_scenarios_for_which_p_vals_were_computed']
    total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign = counts_dict['total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign']
    total_no_with_adjusted_stat_sig = counts_dict['total_no_with_adjusted_stat_sig']

    #Curiosity:
    total_no_with_one_tail_adjusted_stat_sig_when_average_shift_metric_did_not_have_the_wrong_sign = counts_dict['total_no_with_one_tail_adjusted_stat_sig_when_average_shift_metric_did_not_have_the_wrong_sign']
    print(f'total_no_with_one_tail_adjusted_stat_sig_when_average_shift_metric_did_not_have_the_wrong_sign={total_no_with_one_tail_adjusted_stat_sig_when_average_shift_metric_did_not_have_the_wrong_sign}')

    return total_no_scenarios_for_which_p_vals_were_computed,total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign,total_no_with_adjusted_stat_sig

def main():
    print('THE START')

    all_one_tail_global_adjusted_p_vals_files = get_all_one_tail_global_adjusted_p_vals_files(all_top_dirs_containing_raw_p_vals_csvs=[overall_top_public_data_dir,overall_top_syn_data_dir])

    total_no_scenarios_for_which_p_vals_were_computed,total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign,total_no_with_adjusted_stat_sig = count_p_vals_of_interest(all_one_tail_global_adjusted_p_vals_files)

    print(f'total_no_scenarios_for_which_p_vals_were_computed={total_no_scenarios_for_which_p_vals_were_computed}')
    print(f'total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign={total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign}')
    print(f'total_no_with_adjusted_stat_sig={total_no_with_adjusted_stat_sig}')
    print(f'Proportion={round(100*(total_no_with_two_tail_adjusted_stat_sig_when_average_shift_metric_had_the_wrong_sign/total_no_with_adjusted_stat_sig),2)}')
    

    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())


