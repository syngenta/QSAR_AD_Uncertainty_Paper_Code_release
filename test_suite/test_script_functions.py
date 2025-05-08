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
#Copyright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#############################################################
import os,sys
import pandas as pd
from pandas.testing import assert_frame_equal

#------------------
this_dir = os.path.dirname(os.path.abspath(__file__))
top_dir=os.path.dirname(this_dir)
sys.path.append(top_dir)
from modelsADuncertaintyPkg.utils.basic_utils import doubleDefaultDictOfLists
#-------------------
scripts_top_dir = os.path.sep.join([top_dir,'scripts'])
sys.path.append(scripts_top_dir)
#-------------------
from PublicDataModelling.general_purpose.common_globals import assign_new_ids_corresponding_to_inchis
from summary_plots_script_functions import get_plot_input_df,plot_distribution_of_y_vals_per_subset_across_different_x_vals

def test_assign_new_ids_corresponding_to_inchis():
    
    consistent_inchi_col='Fake_InChI_Col'
    
    consistent_id_col='Fake_ID_Col'
    
    fake_train_test_split_1_df = pd.DataFrame({consistent_id_col:[2,4,1,7],consistent_inchi_col:['A','B','D','C']})
    
    fake_train_test_split_2_df = pd.DataFrame({consistent_id_col:[4,2,8,1],consistent_inchi_col:['B','A','C','D']})
    
    updated_df_1 = assign_new_ids_corresponding_to_inchis(unique_df=fake_train_test_split_1_df, consistent_inchi_col=consistent_inchi_col, consistent_id_col=consistent_id_col)
    
    updated_df_2 = assign_new_ids_corresponding_to_inchis(unique_df=fake_train_test_split_2_df, consistent_inchi_col=consistent_inchi_col, consistent_id_col=consistent_id_col)
    
    new_ids_1 = updated_df_1[consistent_id_col].tolist()
    
    inchis_1 = updated_df_1[consistent_inchi_col].tolist()
    
    inchi_to_new_ids_1 = dict(zip(inchis_1,new_ids_1))
    
    new_ids_1.sort()
    
    inchis_1.sort()
    
    new_ids_2 = updated_df_2[consistent_id_col].tolist()
    
    inchis_2 = updated_df_2[consistent_inchi_col].tolist()
    
    inchi_to_new_ids_2 = dict(zip(inchis_2,new_ids_2))
    
    new_ids_2.sort()
    
    inchis_2.sort()
    
    assert new_ids_1 == new_ids_2
    
    assert inchis_1 == inchis_2
    
    assert inchi_to_new_ids_1 == inchi_to_new_ids_2

def test_get_plot_input_df():
    endpoints_to_all_fold_seed_in_and_out_ad_vals = doubleDefaultDictOfLists()
    endpoints_to_all_fold_seed_in_and_out_ad_vals['E1']['Inside'] = [0.8,0.9]
    endpoints_to_all_fold_seed_in_and_out_ad_vals['E1']['Outside'] = [0.5,None]
    endpoints_to_all_fold_seed_in_and_out_ad_vals['E2']['Inside'] = [0.55,0.65]
    endpoints_to_all_fold_seed_in_and_out_ad_vals['E2']['Outside'] = [0.5,0.71]
    
    #========================================
    endpoint_col = 'Endpoint'
    val_col = 'MCC'
    ad_subset_col = 'Subset'
    
    expected_df = pd.DataFrame({endpoint_col:['E1']*4 + ['E2']*4,ad_subset_col:['Inside','Inside','Outside','Outside']*2,val_col:[0.8,0.9,0.5,None,0.55,0.65,0.50,0.71]})
    #========================================

    plot_input_df = get_plot_input_df(endpoints_to_all_fold_seed_in_and_out_ad_vals,endpoint_col,val_col,ad_subset_col)

    assert_frame_equal(expected_df,plot_input_df)

def test_plot_distribution_of_y_vals_per_subset_across_different_x_vals():
    plot_name = 'test_plot_dist_of_vals_vs_EP_grouped_by_ad_subset_label.tiff'

    #========================================
    endpoint_col = 'Endpoint'
    val_col = 'MCC'
    ad_subset_col = 'Subset'
    
    expected_df = pd.DataFrame({endpoint_col:['E1']*4 + ['E2']*4,ad_subset_col:['Inside','Inside','Outside','Outside']*2,val_col:[0.8,0.9,0.5,None,0.55,0.65,0.50,0.71]})
    #========================================

    plot_input_df = expected_df

    x_label = endpoint_col
    y_label = val_col
    subset_label = ad_subset_col

    plot_distribution_of_y_vals_per_subset_across_different_x_vals(plot_name,plot_input_df,x_label,y_label,subset_label)

def test_plot_distribution_of_y_vals_per_subset_across_different_x_vals_middle_ep_has_subset_completely_missing():
    plot_name = 'test_plot_dist_of_vals_vs_EP_grouped_by_ad_subset_label_middle_ep_has_subset_completely_missing.tiff'

    #========================================
    endpoint_col = 'Endpoint'
    val_col = 'MCC'
    ad_subset_col = 'Subset'
    
    expected_df = pd.DataFrame({endpoint_col:['E0']*4 + ['E1']*4 + ['E2']*4,ad_subset_col:['Inside','Inside','Outside','Outside']*3,val_col:[0.25,0.35,0.45,0.50,0.8,0.9,None,None,0.55,0.65,0.50,0.71]})
    #========================================

    plot_input_df = expected_df

    x_label = endpoint_col
    y_label = val_col
    subset_label = ad_subset_col

    plot_distribution_of_y_vals_per_subset_across_different_x_vals(plot_name,plot_input_df,x_label,y_label,subset_label)

def test_plot_distribution_of_y_vals_per_subset_across_different_x_vals_filter_x_with_missing_subset():
    plot_name = 'test_plot_dist_of_vals_vs_EP_grouped_by_ad_subset_label_filter_eps_with_missing_subset.tiff'

    #========================================
    endpoint_col = 'Endpoint'
    val_col = 'MCC'
    ad_subset_col = 'Subset'
    
    expected_df = pd.DataFrame({endpoint_col:['E0']*4 + ['E1']*4 + ['E2']*4,ad_subset_col:['Inside','Inside','Outside','Outside']*3,val_col:[0.25,0.35,0.45,0.50,0.8,0.9,None,None,0.55,0.65,0.50,0.71]})
    #========================================

    plot_input_df = expected_df

    x_label = endpoint_col
    y_label = val_col
    subset_label = ad_subset_col

    plot_distribution_of_y_vals_per_subset_across_different_x_vals(plot_name,plot_input_df,x_label,y_label,subset_label,should_filter_x_with_no_ys_for_at_least_one_subset=True)


