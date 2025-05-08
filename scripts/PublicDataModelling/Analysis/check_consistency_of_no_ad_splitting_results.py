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

this_dir = os.path.dirname(os.path.abspath(__file__))
top_scripts_dir = os.path.dirname(os.path.dirname(this_dir))
sys.path.append(top_scripts_dir)
from consistent_parameters_for_all_modelling_runs import classification_stats_in_desired_order,regression_stats_in_desired_order
data_dir = os.path.sep.join([os.path.dirname(os.path.dirname(os.path.dirname(top_scripts_dir))),'PublicData'])
merged_res_dir = os.path.sep.join([data_dir,'ModellingResSummary'])

scenario_cols = ['Endpoint', 'Test Set Name (ignoring fold if applicable)', 'Fold (if applicable)', 'Random seed', 'Modelling Algorithm', 
'AD Method', 'AD Subset']

def add_scenario_col(row,scenario_cols):
    row['scenario'] = '_'.join([str(row[c]) for c in scenario_cols])
    return row


for modelling_type in ['classification','regression']:
    if 'classification' == modelling_type:
        df = pd.read_csv(os.path.sep.join([merged_res_dir,'Exemplar_Endpoints_Classification_Stats.csv']))
        no_res = 3
        metrics = classification_stats_in_desired_order
    elif 'regression' == modelling_type:
        df = pd.read_csv(os.path.sep.join([merged_res_dir,'Exemplar_Endpoints_Regression_Stats.csv']))
        no_res = 7
        metrics = regression_stats_in_desired_order
    else:
        raise Exception(f'modelling_type={modelling_type}')
    
    df = df.apply(add_scenario_col,axis=1,args=(scenario_cols,))
    all_scenarios = df['scenario'].unique().tolist()

    for scenario in all_scenarios:
        if 'All' == scenario.split('_')[-1]:
            print(f'Scenario={scenario}')
            sub_df = df[df['scenario'].isin([scenario])]
        else:
            continue
        for m in metrics:
            print(f'Metric={m}')
            vals = sub_df[m].tolist()

            assert no_res == len(vals),len(vals)
            #some degree of inconsistency can be expected for the same floating point calculations, mayb run on different machines on the cluster at different times
            #https://softwareengineering.stackexchange.com/questions/433322/should-i-check-floating-point-values-in-a-unit-test
            if not all([pd.isna(v) for v in vals]): assert 1 == len(set([round(v,13) for v in vals])),vals
        




