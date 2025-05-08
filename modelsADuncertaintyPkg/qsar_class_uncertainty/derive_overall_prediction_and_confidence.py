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
#Copyright (c) Syngenta 2020
#Contact richard.marchese_robinson@syngenta.com
#######################
import re
from operator import itemgetter

def basicChecks(p_class_i,p):
    assert re.match('(^p_)',p_class_i),p_class_i
    assert type(0.5) == type(p),str(type(p))
    assert p <= 1,str(p)
    assert p>= 0,str(p)

def deriveFinalPredictionAndConfidence(all_classes_probs_dict):
    
    #------------------------------------------
    for p_class_i in all_classes_probs_dict.keys():
        p = all_classes_probs_dict[p_class_i]
        basicChecks(p_class_i,p)
    #------------------------------------------
    
    raw_Prediction,Prediction_Confidence = sorted(all_classes_probs_dict.items(),key=itemgetter(1),reverse=True)[0]
    
    Prediction = raw_Prediction.split('p_')[1]
    
    return Prediction,Prediction_Confidence
