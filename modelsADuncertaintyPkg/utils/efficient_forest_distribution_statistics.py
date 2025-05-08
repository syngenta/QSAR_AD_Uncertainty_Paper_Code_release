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
###################################################
#Copyright (c) 2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
#This file was adapted from https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py
#The key changes were as follows:
#(i)to update _return_std(...) to only return the standard deviation of tree predictions, rather than compute a variance in the predictions which took into account the variance in the leaf node values;
#(ii)to extend this to allow other tree distribution statistics to be computed
#(iii)to allow this to work with RandomForestClassifier as well
####################################################
# BSD 3-Clause License

# Copyright (c) 2016-2020 The scikit-optimize developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
   # list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
   # this list of conditions and the following disclaimer in the documentation
   # and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
   # contributors may be used to endorse or promote products derived from
   # this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################

import numpy as np

def get_tree_probs_for_class_of_interest(X,tree,class_of_interest):
    all_classes_in_order = tree.classes_.tolist()

    class_of_interest_index = all_classes_in_order.index(class_of_interest)

    probs_for_all_classes = tree.predict_proba(X)

    return probs_for_all_classes[:,class_of_interest_index]

def _return_random_forest_tree_predictions(X,trees,predict_class_probabilities=False,class_of_interest=None):
    '''
    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or RandomForestClassifier.
    '''
    
    if not predict_class_probabilities:
        all_tree_preds_for_all_rows_in_X = np.array([tree.predict(X) for tree in trees])
    else:
        all_tree_probs_for_class_of_interest_for_all_rows_in_X = np.array([get_tree_probs_for_class_of_interest(X,tree,class_of_interest) for tree in trees])
        all_tree_preds_for_all_rows_in_X = all_tree_probs_for_class_of_interest_for_all_rows_in_X


    assert all_tree_preds_for_all_rows_in_X.shape[1] == X.shape[0]
    assert all_tree_preds_for_all_rows_in_X.shape[0] == len(trees)
    

    return all_tree_preds_for_all_rows_in_X

def _return_std(X, trees):
    """
    Returns standard deviation of the predictions of individual trees

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    

    Returns
    -------
    std : array-like, shape=(n_samples,)
    

    """
    
    all_tree_preds_for_all_rows_in_X = _return_random_forest_tree_predictions(X,trees)
    
    std = np.std(all_tree_preds_for_all_rows_in_X,axis=0)

    assert len(std) == X.shape[0]
    
    return std


