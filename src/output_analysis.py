"""
Implements methods which calculate confusion matrix statistics and related
metrics to allow for detailed analysis of the performance of NSB models.

More information about the statistics calculated within may be found at
[wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
"""


import subprocess
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = 'test'

    binary_analysis(model_name)


def binary_analysis(model_name):
    """
    Loads the outputs of a NSB model, which contain the input data, training
    targets, and model predictions.

    Binarizes all three using the mean of each as a threshold.

    Calculates several confusion matrix statistics and logs relevant
    information to a text file for later consumption.

    Generates a mp4 video displaying the original and binarized versions of the
    model output to allow for visual inspection.

    Args:
        model_name (str) Identifier associated with a trained model
    """
    with np.load(f'../data/ModelOutputs/{model_name}_outputs.npz') as data:
        output_data = data['X'], data['Y'], data['P']

    log_file = open(f'../visualizations/Confusion/{model_name}_binary_analysis.log', 'a')

    log_file.write(f'Start: {datetime.now()}\n\n')

    output_data = [np.squeeze(x) for x in output_data]

    bin_data = [binarize(x) for x in output_data]

    results = defaultdict(list)

    plotData = [output_data, bin_data]

    for i in range(len(output_data[0])):
        results['tp'].append(true_positives(bin_data[1][i], bin_data[2][i]))
        results['tn'].append(true_negatives(bin_data[1][i], bin_data[2][i]))
        results['fp'].append(false_positives(bin_data[1][i], bin_data[2][i]))
        results['fn'].append(false_negatives(bin_data[1][i], bin_data[2][i]))
        results['accuracy'].append(accuracy_score(results['tp'][i],
                                                  results['tn'][i],
                                                  results['fp'][i],
                                                  results['fn'][i]))
        results['f1'].append(f1_score(results['tp'][i],
                                      results['tn'][i],
                                      results['fp'][i],
                                      results['fn'][i]))
        results['matthews corr'].append(matthews_correlation(results['tp'][i],
                                                             results['tn'][i],
                                                             results['fp'][i],
                                                             results['fn'][i]))

        # Plot tiled images to allow for visual inspection.
        fig, ax = plt.subplots(2, 3)

        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                ax[j, k].imshow(plotData[j][k][i],
                                cmap=plt.get_cmap('Greys'),
                                interpolation="nearest")

                # Don't draw the axes
                ax[j, k].set_frame_on(False)

                # Remove the ticks
                ax[j, k].set_xticks([])
                ax[j, k].set_yticks([])

                # Create labels and titles
                if not j and not k:
                    ax[j, k].set_ylabel('Original\nImages',
                                        labelpad=30,
                                        rotation=0)
                    ax[j, k].set_title('Input')
                elif not j and k == 1:
                    ax[j, k].set_title('Tracer')
                elif not j and k == 2:
                    ax[j, k].set_title('Output')
                elif j == 1 and not k:
                    ax[j, k].set_ylabel('Binarized\nImages',
                                        labelpad=30,
                                        rotation=0)
                else:
                    pass

        fig.savefig('slice_{0:04d}.png'.format(i))
        plt.close(fig)

    stats = pd.DataFrame(results).describe()
    stats = stats.loc[stats.index != 'count']
    stats = stats[['tp',
                   'tn',
                   'fp',
                   'fn',
                   'accuracy',
                   'f1',
                   'matthews corr']]

    log_file.write('Samples analyzed: {}\n\n'.format(len(output_data[0])))
    log_file.write('{}\n\n\n'.format(stats))
    log_file.close()

    subprocess.run('ffmpeg -framerate 30 -i slice_%04d.png ../visualizations/Videos/{}_binary_analysis.mp4'.format(model_name), shell=True)
    subprocess.run('rm slice*', shell=True)


def binarize(data, threshold=None):
    if not threshold:
        threshold = np.mean(data)

    return np.where(data > threshold,
                    np.ones(data.shape),
                    np.zeros(data.shape))


def true_positives(y_true, y_pred):
    """
    Args:
        y_true (numpy.ndarray) Binary target values.

        y_pred (numpy.ndarray) Binary predictions.

    Returns:
        int in [0, &infin)
    """
    return np.sum(np.logical_and(y_true == 1, y_pred == 1))


def true_negatives(y_true, y_pred):
    """
    Args:
        y_true (numpy.ndarray) Binary target values.

        y_pred (numpy.ndarray) Binary predictions.

    Returns:
        int in [0, &infin)
    """
    return np.sum(np.logical_and(y_true == 0, y_pred == 0))


def false_positives(y_true, y_pred):
    """
    Args:
        y_true (numpy.ndarray) Binary target values.

        y_pred (numpy.ndarray) Binary predictions.

    Returns:
        int in [0, &infin)
    """
    return np.sum(np.logical_and(y_true == 0, y_pred == 1))


def false_negatives(y_true, y_pred):
    """
    Args:
        y_true (numpy.ndarray) Binary target values.

        y_pred (numpy.ndarray) Binary predictions.

    Returns:
        int in [0, &infin)
    """
    return np.sum(np.logical_and(y_true == 1, y_pred == 0))


def recall(tp, fn):
    """
    Calculates the recall of a binary classifier.
    Recall is sometimes referred to as sensitivity, hit rate, or true positive
    rate.

    Args:
        tp (int) Number of true positives.

        fn (int) Number of false negatives

    Returns:float in range [0, 1])
    """
    return 1. * tp / (tp + fn + np.finfo(float).eps)


def specificity(tn, fp):
    """
    Calculates the specificity of a binary classifier.
    Specificity is sometimes referred to as the true negative rate.

    Args:
        tp (int) Number of true negatives.

        fn (int) Number of false positives.

    Returns:float in range [0, 1])
    """
    return 1. * tn / (tn + fp + np.finfo(float).eps)


def precision(tp, fp):
    """
    Calculates the precision of a binary classifier.
    Precison is sometimes referred to as the positive predictive value.

    Args:
        tp (int) Number of true positives.

        fp (int) Number of false positives.

    Returns:float in range [0, 1])
    """
    return 1. * tp / (tp + fp + np.finfo(float).eps)


def negative_predictive_value(tn, fn):
    """
    Calculates the negative predictive value of a binary classifier.

    Args:
        tn (int) Number of true negatives.

        fn (int) Number of false negatives.

    Returns:float in range [0, 1])
    """
    return 1. * tn / (tn + fn + np.finfo(float).eps)


def accuracy_score(tp, tn, fp, fn):
    """
    Calculates the accuracy of a binary classifier.
    Accuracy is sometimes referred to as the Rand index.

    Args:
        tp (int) Number of true positives.

        tn (int) Number of true negatives.

        fp (int) Number of false positives.

        fn (int) Number of false negatives.

    Returns:
        float in range [0, 1]
    """
    return 1. * (tp + tn) / (tp + tn + fp + fn + np.finfo(float).eps)


def f1_score(tp, tn, fp, fn):
    """
    Calculates the F1 score of a binary classifier.

    Args:
        tp (int) Number of true positives.

        tn (int) Number of true negatives.

        fp (int) Number of false positives.

        fn (int) Number of false negatives.

    Returns:
        float in range [0, 1]
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2. * p * r / (p + r + np.finfo(float).eps)


def matthews_correlation(tp, tn, fp, fn):
    """
    Calculates the Matthews correlation coefficient of a binary classifier.
    The MCC considers all of the elements of the confusion matrix (unlike the
    F1 score which fails to consider the number of true negatives). A MCC value
    of +1 represents perfect prediction, 0 represents prediction which is no
    better than random guessing, and -1 represents completely incorrect
    predictions.

    Args:
        tp (int) Number of true positives.

        tn (int) Number of true negatives.

        fp (int) Number of false positives.

        fn (int) Number of false negatives.

    Returns:
        float in range [-1, 1]
    """
    return ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + np.finfo(float).eps)


def informedness(tp, tn, fp, fn):
    """
    Calculates the informedness of a binary classifier.
    Informedness is a component of the Matthews correlation coefficient which
    corresponds with information flow.

    Args:
        tp (int) Number of true positives.

        tn (int) Number of true negatives.

        fp (int) Number of false positives.

        fn (int) Number of false negatives.

    Returns:
        float in range [-1, 1]
    """
    return recall(tp, fn) + specificity(tn, fp) - 1.


def markedness(tp, tn, fp, fn):
    """
    Calculates the markedness of a binary classifier.
    Markedness is a component of the Matthews correlation coefficient which
    corresponds with information flow and is the inverse of informedness.

    Args:
        tp (int) Number of true positives.

        tn (int) Number of true negatives.

        fp (int) Number of false positives.

        fn (int) Number of false negatives.

    Returns:
        float in range [-1, 1]
    """
    return precision(tp, fp) + negative_predictive_value(tn, fn) - 1.


if __name__ == '__main__':
    main()
