# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:35:28 2019

@author: User
"""

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.utils.multiclass import unique_labels
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

# copied from documentation

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def model_classification_report(y_true, y_pred, labels):
    labels = labels.astype('U').tolist()
    return classification_report(y_true, y_pred, target_names=labels)

def evaluate_continuous_data(y_true, y_pred):
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_true, y_pred))
    # Pearson correlation
    print('Pearson correlation coefficient:', pearsonr(y_true, y_pred))

def save_predictions(model_name, x_test_cv, x_test_job, y_true, y_pred):
    df = pd.DataFrame({'x_test_cv': x_test_cv,
                       'x_test_job' : x_test_job,
                       'y_test': y_true,
                       'y_pred': y_pred})
    # there are still errors here
    filename = 'r\'./predictions/'+ model_name + '_' + datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S") + '.csv\''
    df.to_csv(filename, index=False, header=False)
    