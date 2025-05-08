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
#################################################
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#This code was originally written by Zied Hosni [as part of extraFunctions.py] as part of a Syngenta fundeded collaboration with the University of Sheffield. Adaptations were made by richard.marchese_robinson (RMR), e.g. moving Tanimoto similarity into another package function (compute_tanimoto_similarity(...))
#################################################
import pandas as pd
from ..CheminformaticsUtils.similarity import compute_tanimoto_similarity
from ..CheminformaticsUtils.chem_data_parsing_utils import check_fps_df

def TanimotoSplitting(fps_train, fps_test, threshold, test_ids):
    fps_train_df = pd.DataFrame(fps_train.values, columns=['fps train'])
    fps_test_df = pd.DataFrame(fps_test.values, columns=['fps test'])
    test_id_ad_status_dict = coreTanimotoSplitting(fps_train_df, fps_test_df, threshold, test_ids)
    return (test_id_ad_status_dict)

def coreTanimotoSplitting(fps_train, fps_test, threshold,test_ids_in_same_order_as_fps_test=None): # RMR edit: including test_ids_in_same_order_as_fps_test option
    
    #=========================
    check_fps_df(fps_train)
    check_fps_df(fps_test)
    #=========================
    
    test_id_ad_status_dict = {}
    
    for index, fp_test in fps_test.iterrows():  # RMR edit: including test_ids_in_same_order_as_fps_test option
        if test_ids_in_same_order_as_fps_test is None:  # RMR edit: including test_ids_in_same_order_as_fps_test option
            id = index  # RMR edit: including test_ids_in_same_order_as_fps_test option
        else:  # RMR edit: including test_ids_in_same_order_as_fps_test option
            id = test_ids_in_same_order_as_fps_test.iloc[
                index]  # RMR edit: including test_ids_in_same_order_as_fps_test option

        test_id_ad_status_dict[id] = {}
        test_id_ad_status_dict[id]['InsideAD'] = False
        for i, fp_train in fps_train.iterrows():
            sims_dis = 1 - compute_tanimoto_similarity(fp_test[0], fp_train[0])
            assert sims_dis >= 0 and sims_dis <= 1, sims_dis
            if sims_dis < threshold:
                test_id_ad_status_dict[id]['InsideAD'] = True
                break
    return (test_id_ad_status_dict)
