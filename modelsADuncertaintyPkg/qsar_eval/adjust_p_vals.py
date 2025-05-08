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
#Copyright (c) 2023-2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#########################################################
import os,re
import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests

def getAdjustedPValues(p_vals_list,sig_level_perc=5,p_val_adjustment_method='fdr_by',skip_missing=True):
    
    sig_level_fraction = sig_level_perc/100
    
    if not skip_missing:
        res = multipletests(pvals=np.array(p_vals_list),alpha=sig_level_fraction,method=p_val_adjustment_method)
        corrected_p_vals_list = res[1].tolist()
    else:
        #mask = [pd.isna(v) for v in p_vals_list]
        #res = multipletests(pvals=np.ma.array(p_vals_list,mask=mask),alpha=sig_level_fraction,method=p_val_adjustment_method)
        p_vals_series = pd.Series(p_vals_list)
        filtered_series = p_vals_series.dropna(inplace=False)
        if not 0 == len(filtered_series):
            res = multipletests(pvals=np.array(filtered_series),alpha=sig_level_fraction,method=p_val_adjustment_method)
            corrected_filtered_series = pd.Series(res[1],index=filtered_series.index)
            p_vals_series.update(corrected_filtered_series)
            corrected_p_vals_list = p_vals_series.tolist()
        else:
            corrected_p_vals_list = p_vals_list
   
    
    return corrected_p_vals_list

def addAdjustedPValues(out_dir,p_vals_table,sig_level_perc,raw_p_vals_col='Shift-Metric P-value',adj_p_vals_col='Adjusted P-value',p_val_adjustment_method='fdr_by',adjusted_p_vals_table_name=None):
    ##########################################
    #This function will just append the column 'Adjusted P-value' (default name) to the end of p_vals_table
    #The adjustment is required for a set of p-values corresponding to a family of hypotheses, to control for finding more statistically significant differences than would be expected at the specified significance level simply by virtue of having tested a large number of scenarios. This is relevant when any one of the significant differences for a given family might support a more general conclusion, e.g. this applicability domain technique tends to produce statistically significant differences in the metrics obtained inside and outside the domain.
    ############################################
    
    df = pd.read_csv(os.path.sep.join([out_dir,p_vals_table]))
    
    p_vals_list = df[raw_p_vals_col].tolist()
    
    corrected_p_vals_list = getAdjustedPValues(p_vals_list,sig_level_perc,p_val_adjustment_method)
    
    df.insert(df.shape[1],adj_p_vals_col,pd.Series(corrected_p_vals_list),allow_duplicates=True)
    
    if adjusted_p_vals_table_name is None:
        adjusted_p_vals_table_name = re.sub('(\.csv$)','_AdjustedPVals.csv',p_vals_table)
        assert not adjusted_p_vals_table_name == p_vals_table,"{} must be a csv!".format(p_vals_table)
    
    df.to_csv(os.path.sep.join([out_dir,adjusted_p_vals_table_name]),index=False)
