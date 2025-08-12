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


dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
top_scripts_dir = os.path.dirname(os.path.dirname(dir_of_this_script))
top_code_dir = os.path.dirname(top_scripts_dir)

sys.path.append(top_code_dir)
from modelsADuncertaintyPkg.utils.basic_utils import neverEndingDefaultDict

data_dir = os.path.dirname(os.path.dirname(top_code_dir))
dir_with_res_to_analyse = os.path.sep.join([data_dir,'PublicData','AllMergedStats'])

dict_of_applicable_metrics_to_lower_bounds_of_interest = {'R2':0,'Pearson coefficient':0,'Spearman coefficient':0,'Spearman coefficient (PIs vs. residuals)':0,'Validity':0.68,'Balanced Accuracy':0.50,'MCC':0,'AUC':0.5,'Kappa':0,'R2 (cal)':0,'Pearson coefficient (cal)':0,'Spearman coefficient (cal)':0}


dict_of_details = neverEndingDefaultDict()
non_rand_class_test_sets = ['Tox21Score','Tox21Test','holdout','update1']
applicable_class_pred_metrics = ['Balanced Accuracy','MCC','AUC','Kappa']
applicable_class_uncert_metrics = ['R2 (cal)','Pearson coefficient (cal)','Spearman coefficient (cal)']

non_rand_reg_test_sets = ['ivot']
applicable_reg_pred_metrics = ['R2','Pearson coefficient','Spearman coefficient']
applicable_reg_uncert_metrics = ['Spearman coefficient (PIs vs. residuals)','Validity']

dict_of_details['Classification']['Classification prediction performance (non-random test sets)']['test_set_names_of_interest'] = non_rand_class_test_sets
dict_of_details['Classification']['Classification prediction performance (non-random test sets)']['metric_names_of_interest'] = applicable_class_pred_metrics

dict_of_details['Classification']['Classification uncertainty performance (non-random test sets)']['test_set_names_of_interest'] = non_rand_class_test_sets
dict_of_details['Classification']['Classification uncertainty performance (non-random test sets)']['metric_names_of_interest'] = applicable_class_uncert_metrics


dict_of_details['Regression']['Regression prediction performance (non-random test sets)']['test_set_names_of_interest'] = non_rand_reg_test_sets
dict_of_details['Regression']['Regression prediction performance (non-random test sets)']['metric_names_of_interest'] = applicable_reg_pred_metrics

dict_of_details['Regression']['Regression uncertainty performance (non-random test sets)']['test_set_names_of_interest'] = non_rand_reg_test_sets
dict_of_details['Regression']['Regression uncertainty performance (non-random test sets)']['metric_names_of_interest'] = applicable_reg_uncert_metrics

def analyse_this_situation(raw_stats_df,results_subset_of_interest_name,ad_subset,test_set_names_of_interest,metric_names_of_interest,dict_of_applicable_metrics_to_lower_bounds_of_interest):
    
    print('='*50)
    print(f'Analysing this situation : {results_subset_of_interest_name} [AD subset = {ad_subset}]')

    subset_df = raw_stats_df[raw_stats_df['AD Subset'].isin([ad_subset])]
    subset_df = subset_df[subset_df['Test Set Name (ignoring fold if applicable)'].isin(test_set_names_of_interest)]

    for metric_name in metric_names_of_interest:
        lower_bound = dict_of_applicable_metrics_to_lower_bounds_of_interest[metric_name]

        relevant_subset_df = subset_df[~subset_df[metric_name].isna()]

        denominator = relevant_subset_df.shape[0]

        if not 'Validity' == metric_name:
            numerator = relevant_subset_df[relevant_subset_df[metric_name] > lower_bound].shape[0]
        else:
            numerator = relevant_subset_df[relevant_subset_df[metric_name] >= lower_bound].shape[0]
        
        perc = round(100*(numerator/denominator),0)
        
        print(f'{metric_name} ({perc}%)')
        print(f'fraction: {metric_name} ({numerator}/{denominator})')


    print('='*50)

def main():

    print('THE START')

    for modelling_type in ['Classification','Regression']:
        raw_stats_csv = os.path.sep.join([dir_with_res_to_analyse,f'All_Endpoints_{modelling_type}_Stats.csv'])

        raw_stats_df = pd.read_csv(raw_stats_csv)

        for results_subset_of_interest_name in dict_of_details[modelling_type].keys():
            test_set_names_of_interest = dict_of_details[modelling_type][results_subset_of_interest_name]['test_set_names_of_interest']
            metric_names_of_interest = dict_of_details[modelling_type][results_subset_of_interest_name]['metric_names_of_interest']



            for ad_subset in ['Inside','Outside']:

                analyse_this_situation(raw_stats_df,results_subset_of_interest_name,ad_subset,test_set_names_of_interest,metric_names_of_interest,dict_of_applicable_metrics_to_lower_bounds_of_interest)
                
               




    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())