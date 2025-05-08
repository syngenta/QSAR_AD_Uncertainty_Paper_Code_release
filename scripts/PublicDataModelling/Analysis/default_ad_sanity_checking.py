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
####################################################
import pandas as pd
import os,sys
from collections import defaultdict
from functools import partial

this_dir = os.path.dirname(os.path.abspath(__file__))
top_scripts_dir = os.path.dirname(os.path.dirname(this_dir))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import regression_metric_to_expected_sub_1_minus_sub_2_sign,classifcation_metric_to_expected_sub_1_minus_sub_2_sign
from PublicDataModelling.general_purpose.common_globals import exemplar_ChEMBL_endpoints,exemplar_Tox21_endpoints,exemplar_Wang_endpoints
pkg_dir = os.path.dirname(top_scripts_dir)
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.basic_utils import createOrReplaceDir

from AD_ranking import filter_to_only_keep_default_AD_params_results

data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(os.path.dirname(top_scripts_dir))),'PublicData'])
results_to_analyse_top_dir = os.path.sep.join([data_dir,'ADWA'])
out_dir = os.path.sep.join([data_dir,'Default_AD_Sanity_Check'])
merged_res_dir = os.path.sep.join([data_dir,'ModellingResSummary'])

default_ad_method = 'UNC'

def filter_irrelevant_test_sets(raw_shift_metrics_df,focus_on_nominal_out_of_ad_test_sets,ds_subdir):
    if focus_on_nominal_out_of_ad_test_sets:
        if 'Morger_ChEMBL' == ds_subdir:
            sub_df = raw_shift_metrics_df[raw_shift_metrics_df['Test Set Name (ignoring fold if applicable)'].isin(['holdout','update1'])]
        elif 'Tox21' == ds_subdir:
            sub_df = raw_shift_metrics_df[raw_shift_metrics_df['Test Set Name (ignoring fold if applicable)'].isin(['Tox21Test','Tox21Score'])]
        elif 'Wang_ChEMBL' == ds_subdir:
            sub_df = raw_shift_metrics_df[raw_shift_metrics_df['Test Set Name (ignoring fold if applicable)'].isin(['ivot'])]
        else:
            raise Exception(f'ds_subdir={ds_subdir}')
    else:
        sub_df = raw_shift_metrics_df.copy(deep=True)
    
    return sub_df

def compute_relevant_subset(raw_shift_metrics_df,default_ad_method,default_alg,focus_on_nominal_out_of_ad_test_sets,ds_subdir):

    sub_df = filter_irrelevant_test_sets(raw_shift_metrics_df,focus_on_nominal_out_of_ad_test_sets,ds_subdir)
    
    if not default_alg is None:
        sub_df = sub_df[sub_df['Modelling Algorithm'].isin([default_alg])]
    
    sub_df = sub_df[sub_df['AD Method'].isin([default_ad_method])]

    return sub_df

def count_blanks(sub_df,metric):
    return sub_df[metric].isna().sum()

def has_expected_sign(expected_sign,v):
    if (v*expected_sign) > 0:
        return True
    else:
        return False

def compute_fraction_with_expected_sign(sub_df,metric,expected_sign):
    non_blank_subset_df = sub_df[~sub_df[metric].isna()]

    values = non_blank_subset_df[metric].tolist()

    no_with_expected_sign = len(list(filter(partial(has_expected_sign,expected_sign),values)))

    return f'({no_with_expected_sign} / {len(values)})'

def analyse_correct_signs_of_raw_shift_metrics(raw_shift_metrics_df,default_ad_method,ds_subdir,out_dir,default_alg,metric_to_expected_sign,focus_on_nominal_out_of_ad_test_sets):

    sub_df = compute_relevant_subset(raw_shift_metrics_df,default_ad_method,default_alg,focus_on_nominal_out_of_ad_test_sets,ds_subdir)

    metric_findings = defaultdict(list)

    for metric in metric_to_expected_sign.keys():

        expected_sign = metric_to_expected_sign[metric]

        metric_findings['metric'].append(metric)

        metric_findings['blanks'].append(count_blanks(sub_df,f'{metric} Difference'))

        metric_findings['fraction with expected sign'].append(compute_fraction_with_expected_sign(sub_df,f'{metric} Difference',expected_sign))


    out_df = pd.DataFrame(metric_findings)

    out_file = os.path.sep.join([out_dir,f'{ds_subdir}_AD={default_ad_method}_alg={default_alg}_only_nominal_out_of_ad={focus_on_nominal_out_of_ad_test_sets}.csv'])

    out_df.to_csv(out_file,index=False)

def raw_shift_metrics_analysis():
    

    createOrReplaceDir(out_dir)

    for ds_subdir in ['Morger_ChEMBL','Tox21','Wang_ChEMBL']:

        if ds_subdir in ['Morger_ChEMBL','Tox21']:
            default_alg = 'CVAP'
            metric_to_expected_sign = classifcation_metric_to_expected_sub_1_minus_sub_2_sign
        elif ds_subdir in ['Wang_ChEMBL']:
            default_alg = 'ACP'
            metric_to_expected_sign = regression_metric_to_expected_sub_1_minus_sub_2_sign 
        else:
            raise Exception(f'Unrecognised dataset: {ds_subdir}')

        raw_shift_metrics_csv = os.path.sep.join([results_to_analyse_top_dir,ds_subdir,'raw_shift_metrics_df.csv'])

        raw_shift_metrics_df = pd.read_csv(raw_shift_metrics_csv)

        focus_on_nominal_out_of_ad_test_sets = True

        analyse_correct_signs_of_raw_shift_metrics(raw_shift_metrics_df,default_ad_method,ds_subdir,out_dir,default_alg,metric_to_expected_sign,focus_on_nominal_out_of_ad_test_sets)

def get_relevant_subset_of_inside_ad_results(file_to_parse,default_ad_method,default_alg,endpoints_of_interest,ds_subdir,focus_on_nominal_out_of_ad_test_sets):

    df = pd.read_csv(file_to_parse)

    sub_df = filter_to_only_keep_default_AD_params_results(raw_df=df)

    sub_df = sub_df[sub_df['Endpoint'].isin(endpoints_of_interest)]

    sub_df = filter_irrelevant_test_sets(sub_df,focus_on_nominal_out_of_ad_test_sets,ds_subdir)

    if not default_alg is None:
        sub_df = sub_df[sub_df['Modelling Algorithm'].isin([default_alg])]
    
    sub_df = sub_df[sub_df['AD Method'].isin([default_ad_method])]

    sub_df = sub_df[sub_df['AD Subset'].isin(['Inside'])]

    assert not 0 == sub_df.shape[0]

    return sub_df

def is_better_than_random_basline(baseline_val,v):
    if v > baseline_val:
        return True
    else:
        return False

def compute_fraction_with_better_than_random_values(sub_df,metric,metric_to_expected_random_baseline):
    baseline_val = metric_to_expected_random_baseline[metric]

    non_blank_subset_df = sub_df[~sub_df[metric].isna()]

    values = non_blank_subset_df[metric].tolist()

    no_better_than_random = len(list(filter(partial(is_better_than_random_basline,baseline_val),values)))

    return f'({no_better_than_random} / {len(values)})'

def analyse_better_than_random_inside_ad(file_to_parse,default_ad_method,default_alg,endpoints_of_interest,metric_to_expected_random_baseline,ds_subdir,out_dir,focus_on_nominal_out_of_ad_test_sets):

    sub_df = get_relevant_subset_of_inside_ad_results(file_to_parse,default_ad_method,default_alg,endpoints_of_interest,ds_subdir,focus_on_nominal_out_of_ad_test_sets)

    metric_findings = defaultdict(list)

    for metric in metric_to_expected_random_baseline:
        metric_findings['metric'].append(metric)
        metric_findings['blanks'].append(count_blanks(sub_df,metric))

        metric_findings['fraction with better than random values'].append(compute_fraction_with_better_than_random_values(sub_df,metric,metric_to_expected_random_baseline))


    out_df = pd.DataFrame(metric_findings)

    out_file = os.path.sep.join([out_dir,f'{ds_subdir}_InsideBetterThanRandom_AD={default_ad_method}_alg={default_alg}_only_nominal_out_of_ad={focus_on_nominal_out_of_ad_test_sets}.csv'])

    out_df.to_csv(out_file,index=False)

def raw_inside_ad_metrics_analysis():

    for ds_subdir in ['Morger_ChEMBL','Tox21','Wang_ChEMBL']:

        if ds_subdir in ['Morger_ChEMBL','Tox21']:
            default_alg = 'CVAP'
            metric_to_expected_random_baseline = {'Balanced Accuracy':0.50,'MCC':0,'AUC':0.50,'Kappa':0,'R2 (cal)':0,'Pearson coefficient (cal)':0,'Spearman coefficient (cal)':0}
            
            file_to_parse = os.path.sep.join([merged_res_dir,'Exemplar_Endpoints_Classification_Stats.csv'])

            if ds_subdir == 'Morger_ChEMBL':
                endpoints_of_interest = exemplar_ChEMBL_endpoints
            elif ds_subdir == 'Tox21':
                endpoints_of_interest = exemplar_Tox21_endpoints
            else:
                raise Exception(f'ds_subdir={ds_subdir}')
        elif ds_subdir in ['Wang_ChEMBL']:
            default_alg = 'ACP'
            metric_to_expected_random_baseline = {'R2':0,'Pearson coefficient':0,'Spearman coefficient':0,'Spearman coefficient (PIs vs. residuals)':0}
            
            file_to_parse = os.path.sep.join([merged_res_dir,'Exemplar_Endpoints_Regression_Stats.csv'])

            endpoints_of_interest = exemplar_Wang_endpoints
        else:
            raise Exception(f'Unrecognised dataset: {ds_subdir}')
        
        focus_on_nominal_out_of_ad_test_sets = True

        analyse_better_than_random_inside_ad(file_to_parse,default_ad_method,default_alg,endpoints_of_interest,metric_to_expected_random_baseline,ds_subdir,out_dir,focus_on_nominal_out_of_ad_test_sets)
        

def main():
    print('THE START')

    raw_shift_metrics_analysis()

    raw_inside_ad_metrics_analysis()


    print('THE END')

    return 0

if __name__ == '__main__':
    sys.exit(main())
    
