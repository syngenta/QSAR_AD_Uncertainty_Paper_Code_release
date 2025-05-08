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
##############################
#Copright (c) 2024 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import os,sys,re
from collections import defaultdict

ep_type_matched_to_default_AD_uncertainty_methods = defaultdict(dict)
ep_type_matched_to_default_AD_uncertainty_methods['Regression']['AD_method_name'] = 'UNC'
ep_type_matched_to_default_AD_uncertainty_methods['Regression']['uncertainty_method'] = 'ACP'
ep_type_matched_to_default_AD_uncertainty_methods['Classification']['AD_method_name'] = 'UNC'
ep_type_matched_to_default_AD_uncertainty_methods['Classification']['uncertainty_method'] = 'CVAP'
