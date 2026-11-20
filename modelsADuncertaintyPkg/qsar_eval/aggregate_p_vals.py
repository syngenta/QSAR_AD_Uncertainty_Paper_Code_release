#Syngenta Open Source release: This file has been adapted from code developed in the context of a Syngenta funded collaboration with the University of Sheffield: "Improved Estimation of Prediction Uncertainty Leading to Better Decisions in Crop Protection Research". In some cases, this code is a derivative work of other Open Source code. Please see under "If this code was derived from Open Source code, the provenance, copyright and license statements will be reported below" for further details.
#Copyright (c) 2021-2026  Syngenta
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
import numpy as np
import pandas as pd
from .metacp import cauchy_method


def aggregate_p_vals(list_of_p_vals,p_val_aggregation_option="conservative_twice_average",clip_conservative_twice_average=True):
    ####################################
    #When (AD) p-values are obtained via multiple samples, e.g. data splitting or random seeds, we need some way to aggregate the evidence into a single p-value.
    #conservative_twice_average : this approach is based upon an approach described in the statistical literature, e.g.
    # DiCiccio et al. (2020) [https://doi.org/10.1016/j.spl.2020.108865]
    # Choi et al. (2023) [https://doi.org/10.1016/j.spl.2022.109748]
    #cct : Cauchy combination test [https://doi.org/10.1080/01621459.2018.1554485,https://doi.org/10.1186/s12859-025-06126-z]
    ####################################

    #--------------------------------------
    assert isinstance(list_of_p_vals,list),type(list_of_p_vals)
    assert all([(p>=0 and p<=1) for p in list_of_p_vals if not pd.isna(p)]),list_of_p_vals
    #--------------------------------------

    if "conservative_twice_average" == p_val_aggregation_option:
        overall_p = (np.nanmean(np.array(list_of_p_vals,dtype=float),axis=0)*2)
        
        #########################
        #Note: this isn't formally supported by the literature! However, for statistical significance comparisons, this would lead to the same conclusions in practice!
        #########################
        if clip_conservative_twice_average:
            if overall_p > 1:
                overall_p = 1.0
    elif "cct" == p_val_aggregation_option:
        overall_p = cauchy_method(values=[v for v in list_of_p_vals if not pd.isna(v)], data_type="p-values")
    else:
        raise Exception(f'This option for combining p-values is not currently supported: {p_val_aggregation_option}')
    
    return overall_p

