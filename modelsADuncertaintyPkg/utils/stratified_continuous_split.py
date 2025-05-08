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
############################
#Copyright (c) 2023-2024 Syngenta
#contact richard.marchese_robinson [at] syngenta.com
#This file was adapted from the following on 02/11/23: https://github.com/DanilZherebtsov/verstack/blob/master/verstack/stratified_continuous_split.py
#An updated version was copied and then adapted again on 08/10/24
#############################
############################
# MIT License
#
# Copyright (c) 2020-2024 DanilZherebtsov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
############################
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split as split

def estimate_nbins(y):
    """
    Break down target vartiable into bins.

    Args:
        y (pd.Series): stratification target variable.

    Returns:
        bins (array): bins' values.

    """
    if len(y)/10 <= 100:
        nbins = int(len(y)/10)
    else:
        nbins = 100
    bins = np.linspace(min(y), max(y), nbins)
    return bins


def find_neighbors_in_two_lists(keys_with_single_value, list_to_find_neighbors_in):
    '''Iterate over each item in first list to find a pair in the second list'''
    neighbors = []
    for i in keys_with_single_value:
        for j in [x for x in list_to_find_neighbors_in if x != i]:
            if i+1 == j:
                neighbors.append(i)
                neighbors.append(j)
            if i-1 == j:
                neighbors.append(i)
                neighbors.append(j)
    return neighbors

def no_neighbors_found(neighbors):
    '''Check if list is empty'''
    return not neighbors

def find_keys_without_neighbor(neighbors):
    '''Find integers in list without pair (consecutive increment of + or - 1 value) in the same list'''
    no_pair = []
    for i in neighbors:
        if i + 1 in neighbors:
            continue
        elif i - 1 in neighbors:
            continue
        else:
            no_pair.append(i)
    return no_pair

def not_need_further_execution(y_binned_count):
    '''Check if there are bins with single value counts'''
    return 1 not in y_binned_count.values()


def combine_single_valued_bins(y_binned):
    """
    Correct the assigned bins if some bins include a single value (can not be split).

    Find bins with single values and:
        - try to combine them to the nearest neighbors within these single bins
        - combine the ones that do not have neighbors among the single values with
        the rest of the bins.

    Args:
        y_binned (array): original y_binned values.

    Returns:
        y_binned (array): processed y_binned values.

    """
    # count number of records in each bin
    y_binned_count = dict(Counter(y_binned))

    if not_need_further_execution(y_binned_count):
        return y_binned

    # combine the single-valued-bins with nearest neighbors
    keys_with_single_value = [k for k, v in y_binned_count.items() if v == 1]

    # first look for neighbors among other sinle keys
    neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, keys_with_single_value)
    if no_neighbors_found(neighbors1):
        # then look for neighbors among other available keys
        neighbors1 = find_neighbors_in_two_lists(keys_with_single_value, y_binned_count.keys())
    # now process keys for which no neighbor was found
    leftover_keys_to_find_neighbors = list(set(keys_with_single_value).difference(neighbors1))
    neighbors2 = find_neighbors_in_two_lists(leftover_keys_to_find_neighbors, y_binned_count.keys())
    neighbors = sorted(list(set(neighbors1 + neighbors2)))
    
    # split neighbors into groups for combining
    # only possible when neighbors are found
    if len(neighbors) > 0:
        splits = int(len(neighbors)/2)
        neighbors = np.array_split(neighbors, splits)
        for group in neighbors:
            val_to_use = group[0]
            for val in group:
                y_binned = np.where(y_binned == val, val_to_use, y_binned)
                keys_with_single_value = [x for x in keys_with_single_value if x != val]
    # --------------------------------------------------------------------------------
    # now conbine the leftover keys_with_single_values with the rest of the bins
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for val in keys_with_single_value:
        nearest = find_nearest([x for x in y_binned if x not in keys_with_single_value], val)
        ix_to_change = np.where(y_binned == val)[0][0]
        y_binned[ix_to_change] = nearest

    return y_binned

def scsplit(*args, stratify, test_size = 0.3, continuous = True, random_state = None):
    """
    Create stratfied splits for based on categoric or continuous column.

    For categoric target stratification raw sklearn is used, for continuous target
    stratification binning of the target variable is performed before split.

    Args:
        *args (pd.DataFrame/pd.Series): one dataframe to split into train, test
            or X, y to split into X_train, X_val, y_train, y_val.
        stratify (pd.Series): column used for stratification. Can be either a
        column inside dataset:
            train, test = scsplit(data, stratify = data['col'],...)
        or a separate pd.Series object:
            X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y).
        test_size (float): test split size. Defaults to 0.3.
        
        continuous (bool): continuous or categoric target variabale. Defaults to True.
        random_state (int): random state value. Defaults to None.

    Returns:
        if a single object is passed for stratification (E.g. 'data'):
            return:
                train (pd.DataFrame): train split
                valid (pd.DataFrame): valid split
        if two objects are passed for stratification (E.g. 'X', 'y'):
            return:
                X_train (pd.DataFrame): train split independent features
                X_val (pd.DataFrame): valid split independent features
                X_train (pd.DataFrame): train split target variable
                X_train (pd.DataFrame): valid split target variable

    """
    
    #----------------
    #RLMR: simplify function calls by only needing to specify test_size and ensuring train_size is always consistent:
    assert isinstance(test_size,float),type(test_size)
    assert test_size > 0 and test_size < 1,test_size
    train_size = 1 - test_size
    #----------------

    if random_state:
        np.random.seed(random_state)
    if len(args) == 2:
        X = args[0]
        y = args[1]
    else:
        X = args[0].drop(stratify.name, axis = 1)
        y = args[0][stratify.name]

    # non continuous stratified split (raw sklearn)
    if not continuous:
        y = np.array(y)
        y = combine_single_valued_bins(y)
        if len(args) == 2:
            X_train, X_val, y_train, y_val = split(X, y,
                                                   stratify = y,
                                                   test_size = test_size if test_size else None,
                                                   train_size = train_size if train_size else None)
            return X_train, X_val, pd.Series(y_train), pd.Series(y_val) #RLMR: I expect y to be a series for regression or classification
        else:
            temp = pd.concat([X, pd.DataFrame(y, columns = [stratify.name])], axis= 1)
            train, val = split(temp,
                                stratify = temp[stratify.name],
                                test_size = test_size if test_size else None,
                                train_size = train_size if train_size else None)
            return train, val
    # ------------------------------------------------------------------------
    # assign continuous target values into bins
    bins = estimate_nbins(y)
    y_binned = np.digitize(y, bins)
    # correct bins if necessary
    y_binned = combine_single_valued_bins(y_binned)

    # split
    if len(args) == 2:
        X_t, X_v, y_t, y_v = split(X, y_binned,
                                   stratify = y_binned,
                                   test_size = test_size if test_size else None,
                                   train_size = train_size if train_size else None)

        try:
            X_train = X.iloc[X_t.index]
            y_train = y.iloc[X_t.index]
            X_val = X.iloc[X_v.index]
            y_val = y.iloc[X_v.index]
        except IndexError as e:
            raise Exception(f'{e}\nReset index of dataframe/Series before applying scsplit')
        return X_train, X_val, y_train, y_val
    else:
        temp = pd.concat([X, pd.DataFrame(y_binned, columns = [stratify.name], index=y.index)], axis= 1)
        tr, te = split(temp,
                       stratify = temp[stratify.name],
                       test_size = test_size if test_size else None,
                       train_size = train_size if train_size else None)
        train = args[0].loc[tr.index]
        test = args[0].loc[te.index]
        return train, test