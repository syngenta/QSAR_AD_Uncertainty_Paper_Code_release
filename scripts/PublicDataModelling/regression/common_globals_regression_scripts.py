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
import os,sys
#==========================================
dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir_of_this_file)))
#==========================================
sys.path.append(pkg_dir)
from modelsADuncertaintyPkg.utils.ML_utils import checkEndpointLists

all_Wang_endpoints = ['A2a', 'ABL1', 'Acetylcholinesterase', 'Aurora-A', 'Cannabinoid', 'Carbonic', 'Caspase', 'Coagulation', 'COX-1', 'COX-2', 'Dihydrofolate', 'Dopamine', 'Ephrin', 'erbB1', 'Estrogen', 'Glucocorticoid', 'Glycogen', 'HERG', 'JAK2', 'LCK', 'Monoamine', 'opioid', 'Vanilloid']

print(f'Datasets for {len(all_Wang_endpoints)} provided by Wang et al.')
##########################################
#Exceptions raised in previous run of generate_Wang_ChEMBL_SMILES_files.py:
#"Exception: err=list index out of range,endpoint=Aurora-A,test_name=ivot,fold=3" 
#"Exception: err=list index out of range,endpoint=erbB1,test_name=ivot,fold=3"
all_Wang_endpoints.remove('Aurora-A')
all_Wang_endpoints.remove('erbB1')
print(f'Retained {len(all_Wang_endpoints)}')
#########################################

exemplar_Wang_endpoints = ['Dopamine', 'COX-1', 'COX-2']

checkEndpointLists(all_eps=all_Wang_endpoints,exemplar_eps=exemplar_Wang_endpoints)

wang_raw_smiles_col = 'smiles'

wang_raw_act_col = 'labels'

no_wang_outer_folds = 5

wang_test_set_names = ['ivit','ivot']

wang_ids_col = 'ID'

wang_top_ds_dir = os.path.sep.join([os.path.dirname(os.path.dirname(pkg_dir)),'PublicData','Wang_ChEMBL'])

def updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_test_predictions,subset_sig_level_2_prediction_intervals):
    
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_ids'] = subset_test_ids
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_y'] = subset_test_y
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_predictions'] = subset_test_predictions
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_sig_level_2_prediction_intervals'] = subset_sig_level_2_prediction_intervals
