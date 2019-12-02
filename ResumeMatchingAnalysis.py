# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:05:41 2019

@author: User
"""

import pandas as pd
import numpy as np

def get_best_preds(x_test, y_test, y_pred):
    """
    Returns a table of the predicted '1' label with corresponding text pair.
    Args:
        x_test: test set sentence pairs; elements are [cv, jobpost]
        y_test: test set labels (encoded 0-4)
        y_pred: predicted labels
    Returns:
        dataframe with sentence pair and labels
    """
    # TODO if no 1 label exists, then go to 2, then 3 and so on
    # gets indices of elements where predicted label is 1
    result = np.where(y_pred == 0)
    print(result)
    cv = x_test[:,0]
    jobpost = x_test[:,1]
    
    # gets elements where predicted label is 1
    cv = np.take(cv, result).tolist()
    jobpost = np.take(jobpost, result).tolist()
    y_test = np.take(y_test, result).tolist()
    
    # return dataframe
    return pd.DataFrame(list(zip(cv, jobpost, y_test)),
                        columns=['CV', 'Jobpost', 'Actual rating'])
        
    