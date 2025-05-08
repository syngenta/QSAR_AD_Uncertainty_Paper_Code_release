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
#######################
#Copyright (c) 2020-2022 Syngenta
#Contact richard.marchese_robinson@syngenta.com
#######################
import os,re,glob
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict

def averagePerfStats(all_folds_combined_stats_files,average_perf_stats_file,ad_status_col):
    all_dfs = [pd.read_csv(f) for f in all_folds_combined_stats_files]
    
    final_df = averagePerfStats_for_all_dfs(all_dfs,ad_status_col)
    
    final_df.to_csv(average_perf_stats_file,index=True)
    

def averagePerfStats_for_all_dfs(all_dfs,ad_status_col):
    
    combined_df = pd.concat(all_dfs)
    
    combined_df_mean_per_ad_status = combined_df.groupby([ad_status_col]).mean()
    
    all_stats_names = combined_df_mean_per_ad_status.columns.values.tolist() #AD status will be row index and other non-numeric columns, such as model descriptions - which should be the same when averaging statistics for a given AD status! - will simply be dropped!
    
    combined_df_mean_per_ad_status = combined_df_mean_per_ad_status.rename(columns=dict(zip(all_stats_names,['%s_mean' % name for name in all_stats_names])))
    
    combined_df_SD_per_ad_status = combined_df.groupby([ad_status_col]).std()
    
    combined_df_SD_per_ad_status = combined_df_SD_per_ad_status.rename(columns=dict(zip(all_stats_names,['%s_sd' % name for name in all_stats_names])))
    
    final_df = pd.concat([combined_df_mean_per_ad_status,combined_df_SD_per_ad_status],axis=1)
    
    final_df = final_df[list(itertools.chain(*[['%s_mean' % v,'%s_sd' % v] for v in all_stats_names]))]
    
    return final_df

def check_averagePerfStats_for_all_dfs():
    examples_dict = defaultdict(dict)
    ############################
    examples_dict[1]['input'] = []
    examples_dict[1]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':[True,False],'BA':[0.80,0.60]}).set_index('AD'))
    examples_dict[1]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':[True,False],'BA':[0.4,0.2]}).set_index('AD'))
    
    examples_dict[1]['ad_status_col'] = 'AD'
    
    examples_dict[1]['expected_output'] = pd.DataFrame({'AD':[False,True],'BA_mean':[0.4,0.6],'BA_sd':[0.28,0.28]}).set_index('AD')
    ############################
    ############################
    examples_dict[2]['input'] = []
    examples_dict[2]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':[False,True],'BA':[0.60,0.80]}).set_index('AD'))
    examples_dict[2]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':[True,False],'BA':[0.4,0.2]}).set_index('AD'))
    
    examples_dict[2]['ad_status_col'] = 'AD'
    
    examples_dict[2]['expected_output'] = pd.DataFrame({'AD':[False,True],'BA_mean':[0.4,0.6],'BA_sd':[0.28,0.28]}).set_index('AD')
    ############################
    ############################
    examples_dict[3]['input'] = []
    examples_dict[3]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':['Yes','No'],'BA':[0.80,0.60]}).set_index('AD'))
    examples_dict[3]['input'].append(pd.DataFrame({'Model':['Ensemble','Ensemble'],'AD':['Yes','No'],'BA':[0.4,0.2]}).set_index('AD'))
    
    examples_dict[3]['ad_status_col'] = 'AD'
    
    examples_dict[3]['expected_output'] = pd.DataFrame({'AD':['No','Yes'],'BA_mean':[0.4,0.6],'BA_sd':[0.28,0.28]}).set_index('AD')
    ############################
    
    for eg in examples_dict.keys():
        print('Checking averagePerfStats_for_all_dfs(...) for example %d' % eg)
        
        output = averagePerfStats_for_all_dfs(examples_dict[eg]['input'],examples_dict[eg]['ad_status_col'])
        
        assert examples_dict[eg]['expected_output'].equals(output.round(2)), "expected_output= %s \n output = %s \n" % (str(examples_dict[eg]['expected_output']),str(output))
        
        print('CHECKED averagePerfStats_for_all_dfs(...) for example %d' % eg)

check_averagePerfStats_for_all_dfs()
