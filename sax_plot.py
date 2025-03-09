#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats as stats
from sax import SAX
from experiment1 import load_ucr_file

def main():
    file_path = "../UCR_TS_Archive_2015/Herring/Herring_TEST"
    labels, series = load_ucr_file(file_path)

    ts = series[0][:128]
    
    w = 8
    a = 3
    x_width = 128//w
    # Color mapping for the alphabet: 'a' black, 'b' light grey, 'c' dark grey
    color_map = {'a': 'black', 'b': 'lightgrey', 'c': 'dimgray'}
    sax_letters = SAX(w=w, a=a, ts=ts).transform()

    breakpoints = SAX.get_breakpoints(a)
    
    fig = plt.figure(figsize=(10,6))
    # Create a gridspec with two columns:
    # Left: density (narrow), Right: main time-series and SAX bars
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.15)

    ax_density = fig.add_subplot(gs[0])
    # Compute a density estimate using a Gaussian KDE
    kde = stats.gaussian_kde(ts)
    ts_vals = np.linspace(np.min(ts), np.max(ts), 100)
    density = kde(ts_vals)
    ax_density.plot(density, ts_vals, color='k')
    ax_density.fill_betweenx(ts_vals, density, color='grey', alpha=0.3)
    ax_density.set_xlabel('')
    ax_density.set_ylabel('')
    ax_density.set_xticks([])
    ax_density.set_yticks([])

    # Main time series plot on right
    ax_main = fig.add_subplot(gs[1])
    time = np.arange(len(ts))
    ax_main.plot(time, ts, color='blue', lw=1)
    ax_main.set_xlabel('')
    ax_main.set_ylabel('')
    ax_main.tick_params(axis='both', which='both', labelsize=10)

    # Add horizontal grey lines for breakpoints across the main plot
    for bp in breakpoints:
        ax_main.axhline(y=bp, color='grey', linestyle='--', lw=0.8)

    bar_height = 0.03  # 3% of the range
    for j, letter in enumerate(sax_letters):
        color = color_map.get(letter, 'gray')
        y = np.mean(ts[j*x_width:(j+1)*x_width])

        ax_main.fill_between([j*x_width, (j+1)*x_width],
                                y, y + bar_height, color=color, edgecolor='black')

        ax_main.text(j * x_width + x_width/2, y + bar_height, letter,
                    color='black', ha='center', va='bottom', fontsize=12)

    ax_main.set_xticklabels([])
    
    plt.suptitle("SAX Representation of a Time Series from the UCR dataset", fontsize=12, y=0.95)
    plt.savefig('sax_plot.png')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
