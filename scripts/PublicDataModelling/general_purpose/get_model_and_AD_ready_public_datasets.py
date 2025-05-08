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
import os,sys,time

scripts_to_run_in_order = []
scripts_to_run_in_order.append('filter_duplicates_plus_retain_InChIs_for_all_datasets.py')
scripts_to_run_in_order.append('filter_cmpds_based_on_additional_rules.py')
scripts_to_run_in_order.append('standardize_filter_via_new_InChIs_calc_FPs.py')
scripts_to_run_in_order.append('rand_split_class_datasets_and_create_csvs_for_all.py')
scripts_to_run_in_order.append('get_all_train_test_splits_ready_for_modelling_and_AD_calc.py')

for script in scripts_to_run_in_order:
    cmd = f'python {script} > {script}.log 2>&1'
    print(f'Running {cmd}')
    start = time.time()
    assert 0 == os.system(cmd)
    end = time.time()
    total_time_in_minutes = (end-start)/60
    print(f'RAN {cmd}')
    print(f'Time taken = {total_time_in_minutes} minutes')
