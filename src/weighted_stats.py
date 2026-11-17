# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Wolfram Tuschewitzki
#
# SPDX-License-Identifier: MIT
"""
Weighted means, medians and quantiles.
"""

import numpy as np
import pandas as pd


def weighted_mean(values, weights, by=None):
    """
    `values` may be an array, Series or DataFrame; `by` groups like pandas'
    `groupby`.
    """
    if not isinstance(values, (pd.Series, pd.DataFrame)):
        values, weights = np.asarray(values), np.asarray(weights)
        return float((values * weights).sum() / weights.sum())

    weighted = values.mul(weights, axis=0)
    if by is None:
        return weighted.sum() / weights.sum()
    return weighted.groupby(by).sum().div(weights.groupby(by).sum(), axis=0)


def weighted_quantiles(values, weights, quantiles):
    """
    Step definition: the smallest value whose cumulative weight share reaches
    each quantile.
    """
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values)
    cumulated = np.cumsum(weights[order]) / weights.sum()
    return [values[order][np.searchsorted(cumulated, q)] for q in quantiles]


def weighted_median(values, weights):
    return weighted_quantiles(values, weights, [0.5])[0]


def interpolated_weighted_quantiles(values, weights, quantiles):
    """
    Interpolated between the weight midpoints of the sorted values.
    """
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    return np.interp(np.asarray(quantiles) * weights.sum(), cumulative, values)
