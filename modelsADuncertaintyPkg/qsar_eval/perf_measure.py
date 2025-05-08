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
#Copyright (c) 2022-2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#The original version of this code file (perf_measure.py) was developed at the Uppsala University and downloaded from https://github.com/pharmbio/SCPRegression [See below for the original copyright and license information.]
#Edits made, where necessary, by Zied Hosni (z.hosni [at] sheffield.ac.uk), whilst working on a Syngenta funded project, and subsequently by Richard Marchese Robinson (richard.marchese_robinson [at] syngenta.com)
####################################################
#Copyright (c) 2019-2022 Uppsala University
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#Kindly cite our paper:
#Gauraha, N. and Spjuth, O.
#Synergy Conformal Prediction for Regression
#Proceedings of the 10th International Conference on Pattern Recognition Applications and Methods. vol 1: ICPRAM, 212-221. (2021). DOI: #10.5220/0010229402120221
#http://dx.doi.org/10.5220/0010229402120221
######################################################

import matplotlib.pyplot as plt
import numpy as np
import sys
import math


# Computes efficiency of a conformal predictor, average width of the intervals
def Efficiency(intervals):
    nrTestCases = len(intervals)

    mean_width = np.mean(abs(intervals[:,0]-intervals[:,1]))

    return mean_width


def ErrorRate(intervals, testLabels):
    if (intervals is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")

    nrTestCases = len(testLabels)

    err = 0

    for j in range(0, nrTestCases):
        if (intervals[j, 0] > testLabels[j]) or \
                (intervals[j, 1] < testLabels[j]):
            err = err + 1

    err = err / nrTestCases

    return err

