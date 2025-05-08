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

all_Tox21_endpoints = ['NR-AR','NR-AR-LBD','NR-AhR','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53','NR-Aromatase']
exemplar_Tox21_endpoints = ['NR-Aromatase', 'NR-AR', 'SR-HSE', 'SR-ARE']

checkEndpointLists(all_eps=all_Tox21_endpoints,exemplar_eps=exemplar_Tox21_endpoints)

exemplar_Tox21_endpoints_2_alt_names = dict(zip(exemplar_Tox21_endpoints,[n.replace('-','_') for n in exemplar_Tox21_endpoints]))

all_ChEMBL_endpoints = ['CHEMBL220','CHEMBL4078','CHEMBL5763','CHEMBL203','CHEMBL206','CHEMBL279','CHEMBL230','CHEMBL340','CHEMBL240','CHEMBL2039','CHEMBL222','CHEMBL228'] #From Table 1 of Morger et al. (2022) [https://www.nature.com/articles/s41598-022-09309-3]

exemplar_ChEMBL_endpoints = ["CHEMBL228", "CHEMBL240", "CHEMBL206","CHEMBL4078"]

checkEndpointLists(all_eps=all_ChEMBL_endpoints,exemplar_eps=exemplar_ChEMBL_endpoints)

tox21_ids_col = "ID"
tox21_smiles_col = "RDKit_SMILES"


chembl_ids_col='molecule_chembl_id'

chembl_smiles_col='smiles'

chembl_act_class_col='activity_class'

chembl_subsets_of_interest = ['train','update1','holdout']

tox21_subsets_of_interest = ['Tox21Train','Tox21Score','Tox21Test']

ds_name_to_train_subset_name = {}
ds_name_to_train_subset_name['Morger_ChEMBL'] = chembl_subsets_of_interest[0]
ds_name_to_train_subset_name['Morger_Tox21'] = tox21_subsets_of_interest[0]

ds_name_to_test_subset_names = {}
ds_name_to_test_subset_names['Morger_ChEMBL'] = chembl_subsets_of_interest[1:]
ds_name_to_test_subset_names['Morger_Tox21'] = tox21_subsets_of_interest[1:]

rand_test_split_fract = 0.2

def updateDictOfRawInsideVsOutsideADResults(dict_of_raw_results,endpoint,test_set_label,rand_seed,AD_method_name,method,AD_subset,subset_test_ids,subset_test_y,subset_probs_for_class_1,subset_predicted_y):
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_ids'] = subset_test_ids
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_test_y'] = subset_test_y
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_probs_for_class_1'] = subset_probs_for_class_1
    dict_of_raw_results[endpoint][test_set_label][rand_seed][AD_method_name][method][AD_subset]['subset_predicted_y'] = subset_predicted_y
