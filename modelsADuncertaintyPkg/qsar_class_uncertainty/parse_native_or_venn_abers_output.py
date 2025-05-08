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
#Copyright (c) Syngenta 2023
#Contact richard.marchese_robinson [at] syngenta.com
#######################

def getProbsForClass1(method,IVAP_test_id_to_pred_class_class_1_prob_p1_p0,CVAP_test_id_to_pred_class_class_1_prob_p1_p0,native_test_id_to_pred_class_class_1_prob,subset_test_ids):
    
    if 'IVAP' == method:
        res_dict = IVAP_test_id_to_pred_class_class_1_prob_p1_p0
    elif 'CVAP' == method:
        res_dict = CVAP_test_id_to_pred_class_class_1_prob_p1_p0
    elif 'Native' == method:
        res_dict = native_test_id_to_pred_class_class_1_prob
    else:
        raise Exception('Unrecognised uncertainty method = {}'.format(method))
    
    subset_probs_for_class_1 = [res_dict[test_id]['Class_1_prob'] for test_id in subset_test_ids]
    
    return subset_probs_for_class_1
