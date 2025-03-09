#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sax import SAX

def load_ucr_file(filepath):
    """Load UCR time series data from a file."""
    data = np.loadtxt(filepath, delimiter=",")
    labels, series = data[:, 0], data[:, 1:]
    return labels, series

def grid_run_tightness_of_lower_bound(series):
    # Calculate tightness for every combination of time series

    w_range = range(2, 9)
    a_range = range(3, 12)
    avg_tightness_scores = np.zeros((len(w_range), len(a_range)))

    print(f'Running {len(w_range) * len(a_range)} combinations of w and a values')
    print(f'Number of time series: {len(series)}')
    n_comparisons = len(w_range) * len(a_range) * len(series) * (len(series) - 1) // 2
    print(f'Total number of comparisons: {n_comparisons}')
    
    with tqdm(total=n_comparisons) as pbar:
        for w in w_range:
            for a in a_range:
                tightness_scores = []
                for i in range(len(series)):
                    for j in range(i + 1, len(series)):
                        ts1 = series[i][:256]  # Clip time series to length 256
                        ts2 = series[j][:256]

                        # Inner loop for w and a values
                        sax1 = SAX(w, a, ts1).transform()
                        sax2 = SAX(w, a, ts2).transform()
                        mindist_score = SAX.mindist(sax1, sax2, a)
                        distance = SAX.euclidean_distance(ts1, ts2)

                        # Handle cases where distance is zero
                        if distance == 0:
                            tightness = 0
                        else:
                            tightness = SAX.tightness_of_lower_bound(mindist_score, distance)

                        tightness_scores.append(tightness)
                        pbar.update(1)

                average_tightness_score = np.mean(tightness_scores)
                avg_tightness_scores[w-2, a-3] = average_tightness_score

    return avg_tightness_scores

if __name__ == '__main__':
    series = []

    for directory in os.listdir('UCR_TS_Archive_2015/'):
        file = os.path.join('UCR_TS_Archive_2015/', directory, directory + '_TRAIN')
        _, result = load_ucr_file(file)
        if len(result[0]) >= 256:
            series.append(result[0])

    tightness_scores = grid_run_tightness_of_lower_bound(series)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    w_range = range(2, 9)
    a_range = range(3, 12)
    w, a = np.meshgrid(w_range, a_range)
    w = w.flatten()
    a = a.flatten()
    tightness_scores_flat = tightness_scores.flatten()

    ax.bar3d(w, a, np.zeros_like(tightness_scores_flat), 0.8, 0.8, tightness_scores_flat, color='white', edgecolor='black', shade=False)
    ax.set_xlabel('Word Size w')
    ax.set_ylabel('Alphabet size a')
    ax.set_zlabel('Tightness of lower bound')

    ax.set_xlim(max(w_range), min(w_range))  # Invert x-axis

    plt.savefig('tightness_of_lower_bound.png')
    plt.show()
