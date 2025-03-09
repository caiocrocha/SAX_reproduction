#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

class SAX(object):
    def __init__(self, w, a, ts):
        self.w = w
        self.a = a
        self.ts = ts
        self.paa_rep = None

    def normalize(self, ts):
        """Normalize the time series to have zero mean and unit variance."""
        return (ts - np.mean(ts)) / np.std(ts)

    def paa(self, ts, w):
        """Reduce dimensionality using Piecewise Aggregate Approximation (PAA)."""
        n = len(ts)
        segment_size = n / w  # Each segment's size in the original series
        paa_rep = np.zeros(w)
        
        for i in range(w):
            start = int(round(i * segment_size))
            end = int(round((i + 1) * segment_size))
            paa_rep[i] = np.mean(ts[start:end])  # Compute mean over each segment
        
        return paa_rep

    @staticmethod
    def get_breakpoints(a):
        """Get breakpoints from a standard normal distribution."""
        return stats.norm.ppf(np.linspace(0, 1, a + 1)[1:-1])

    def transform(self):
        """Convert a time series into a SAX representation."""
        self.ts = self.normalize(self.ts)
        self.paa_rep = self.paa(self.ts, self.w)
        breakpoints = SAX.get_breakpoints(self.a)
        sax_symbols = np.digitize(self.paa_rep, breakpoints)
        return ''.join(chr(ord('a') + s) for s in sax_symbols)

    @staticmethod
    def euclidean_distance(ts1, ts2):
        """Compute the Euclidean distance between two time series."""
        return np.sqrt(np.sum((ts1 - ts2) ** 2))

    def paa_distance(self, paa1, paa2, n, w):
        """Lower-bounding approximation of the Euclidean distance using PAA."""
        return np.sqrt(n / w * np.sum((paa1 - paa2) ** 2))

    @staticmethod
    def mindist(sax1, sax2, a):
        """Compute the MINDIST between two SAX representations."""
        breakpoints = SAX.get_breakpoints(a)
        dist_matrix = np.zeros((a, a))
        for i in range(a):
            for j in range(i + 1, a):
                dist_matrix[i, j] = dist_matrix[j, i] = (breakpoints[j - 1] - breakpoints[i]) ** 2
        return np.sqrt(np.sum([dist_matrix[ord(i) - ord('a'), ord(j) - ord('a')] for i, j in zip(sax1, sax2)]))

    @staticmethod
    def tightness_of_lower_bound(mindist_score, distance):
        """Compute the tightness of the lower bound."""
        return mindist_score / distance
