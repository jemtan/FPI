"""
Tools for manipulating sets of variables.
"""

import numpy as np
from keras import backend as K
import tensorflow as tf
import copy

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    res = []
    for variables in zip(*var_seqs):
        res.append(np.mean(variables, axis=0))
    return res

def subtract_vars(var_seq_1, var_seq_2):
    """
    Subtract one variable sequence from another.
    """
    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
    """
    Add two variable sequences.
    """
    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
    """
    Scale a variable sequence.
    """
    return [v * scale for v in var_seq]

def update_aux(var_seq_train,var_seq_aux,var_list,tvar_list):
    """
    Copy aux variables into the train set
    return variable list with
    trainable values from train and 
    auxiliary from aux
    """
    var_seq = [var_train_i if lyr_i in tvar_list else var_aux_i for var_train_i,var_aux_i,lyr_i in zip(var_seq_train,var_seq_aux,var_list)]    
    
    
    return var_seq


