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
#Copright (c) 2023 Syngenta
#Contact richard.marchese_robinson [at] syngenta.com
###############################
import time
import timeit
import os

def basic_time_task(task,end,start,units='minutes'):
    ############################
    #task = string (description)
    #start = time.time()
    #end = time.time()
    ############################

    if 'minutes' == units:
        print(f'Time to {task} = {round((end-start)/60,1)} minutes')
    elif 'seconds' == units:
        print(f'Time to {task} = {round((end-start),1)} seconds')
    else:
        raise Exception(f'Unrecognised time units: {units}')

def times_of_repeated_many_function_calls(function_call_string_or_call,import_statement,how_many=100,repeats=5,collect_garbage=False):
    #uses timeite module: https://docs.python.org/3/library/timeit.html
    #This apparently accepts callables as well as stmts: https://stackoverflow.com/questions/3698230/how-to-pass-current-object-referenceself-to-the-timer-class-of-timeit-module

    if collect_garbage:
        timer = timeit.Timer(stmt=function_call_string_or_call,setup=os.linesep.join([import_statement, 'gc.enable()']))
    else:
        timer = timeit.Timer(stmt=function_call_string_or_call,setup=import_statement)
    
    return timer.repeat(repeat=repeats,number=how_many)
