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
#Copyright (c)  2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
########################
import numpy as np
import pandas as pd


def get_no_instances(input_for_stats):
    ####################
    #input_for_stats would be a sequence comprising any regression or classification model output OR experimentally derived values
    #######################
    if isinstance(input_for_stats,np.ndarray):
        return len(input_for_stats)
    elif isinstance(input_for_stats,pd.Series):
        return len(input_for_stats)
    elif isinstance(input_for_stats,list):
        return len(input_for_stats)
    else:
        raise Exception(f'Unexpected type(input_for_stats)={type(input_for_stats)}')

def size_of_inputs_for_stats_is_big_enough(experi_input,pred_input,limit=2):
    
    e_size = get_no_instances(experi_input)
    p_size = get_no_instances(pred_input)
    
    if not e_size == p_size: raise Exception(f'no. experimentally derived inputs for statistics={e_size} vs. no. model derived inputs={p_size}')
    
    if e_size >= limit:
        return True
    else:
        return False
