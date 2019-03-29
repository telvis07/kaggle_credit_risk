"""
author: Telvis Calhoun

Helper functions
"""
import scipy

import numpy as np
import matplotlib.pyplot as plt


def plot_roc_curve_interp(fpr, tpr, label=None, output_png=None):
    """
    Plot ROC curve

    :param fpr: fpr array from roc_curve()
    :param tpr: tpr array from roc_curve(
    :param label: positive label
    :param output_png: File path to write PNG file

    :return: None
    """
    all_fpr = np.linspace(0, 1, 1001)
    mean_tpr = np.zeros_like(all_fpr)
    mean_tpr += scipy.interp(all_fpr, fpr, tpr)

    plt.plot(all_fpr, mean_tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 0.5, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if output_png:
        plt.savefig(output_png)

