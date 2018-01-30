#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:04:35 2018

@author: mingxing
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def fill_na(X_df):
    X_df['gender'].fillna(2)
    X_df['age_range'].fillna(0)
    return X_df


def compute_positive_rate(X, col, MIN_COUNT):
    ''' MIN_COUNT: we compute the rate for at least MIN_COUNT occurrences,
    since if an item appears only once and it's a repeat buyer,
    we can hardly say the return rate is 100% '''
    # approximately 150 seconds
    group = X.groupby(col)
    group_count = group.count()['label']
    min_count = group_count < MIN_COUNT
    group_count[min_count] = 1.0
    group_posi = group.sum()['label']
    group_posi[min_count] = 0.5
    # return defaultdict(lambda:0.5, [(str(index),
    # group_posi[index]/group_count[index]) for index in group_posi.index])
    return group_posi / group_count


def compute_index(rate, TOP=20):
    LAST = 100 - TOP
    rate_index = list(rate.sort_values(ascending=False).index)
    rate_index = dict(
        zip(rate_index[0:TOP] + rate_index[-LAST:], np.arange(100)))
    return rate_index


def expand_act_log(X_df):
    act_log = []
    for i in range(len(X_df)):
        logs = X_df.loc[i][4].split('#')
        label = X_df.loc[i][5]
        user_id = X_df.loc[i][0]
        merchant_id = X_df.loc[i][3]
        for l in logs:
            act_log.append(
                [user_id] +
                [merchant_id] +
                l.strip().split(':') +
                [label])
    act_log = pd.DataFrame(
        act_log,
        columns=[
            'user_id',
            'merchant_id',
            'item_id',
            'category_id',
            'brand_id',
            'time_stamp',
            'action_type',
            'label'],
        dtype='category')
    return act_log


def log2feature(act_log, rate_brand_index, rate_cate_index, index_dates):
    n = len(act_log)
    p = len(rate_brand_index)
    q = len(rate_cate_index)  # by default p+q it's 200
    # 68=17*4, corresponding to 6 months and 11 days of November, each date
    # has 4 action types
    features = np.zeros((n, p + q + 68))
    brand_keys = rate_brand_index.keys()
    cate_keys = rate_cate_index.keys()
    dates_keys = index_dates.keys()
    for i, log in enumerate(act_log):
        acts = log.split('#')
        for act in acts:
            _, cate, brand, date, act_type = act.strip().split(':')
            if str(brand) in brand_keys:
                features[i][rate_brand_index[str(brand)]] += 1
            if str(cate) in cate_keys:
                features[i][p + rate_cate_index[str(cate)]] += 1
            if str(date) in dates_keys:
                features[i][p +
                            q +
                            index_dates[str(date)] +
                            int(act_type)] += 1
    return features


def generate_dates():
    # we can observe that data begins from May until 11 November
    months = ['05', '06', '07', '08', '09', '10']
    index_dates = {}
    for i, m in enumerate(months):
        for j in range(31):
            index_dates[m + str(j + 1).zfill(2)] = i * 4
    for i in range(11):
        index_dates['11' + str(i + 1).zfill(2)] = (6 + i) * 4
    return index_dates


class FeatureExtractor():
    def __init__(self, MIN_COUNT=6):
        # return rate for each merchant, eg: for a merchant i, the percentage
        # of repeated client
        self.merchant_return_rate = None
        # return rate for each client, eg: for a client i, the percentage of
        # being a repeated client
        self.user_return_rate = None
        # top return rate brands, eg: the first brand is the one which has a
        # highest percentage of repeated client
        self.rate_brand_index = None
        # top return rate categorys, eg: the first category is the one which
        # has a highest percentage of repeated client
        self.rate_cate_index = None
        self.act_log = None  # extract the act log from the raw data
        self.age_dummy = None  # dummy coding age_range
        self.gender_dummy = None  # dummy coding gender_range
        self.index_dates = None  # we count the different action types for
        # different dates minimum counts for a reliable return rate,
        # eg: if one has only 1 count, and it is a repeated client,
        self.MIN_COUNT = MIN_COUNT
    # we can't draw a conclusion that the return rate is 100%, instead we take
    # 50% (equal chance).

    def fit(self, X_df, y=None):
        X_df = fill_na(X_df).reset_index(drop=True)
        X_df['label'] = y
        self.act_log = expand_act_log(X_df)
        self.index_dates = generate_dates()
        rate = compute_positive_rate(X_df, ['merchant_id'], self.MIN_COUNT)
        self.merchant_return_rate = defaultdict(
            lambda: 0.5, [(str(index), rate[index]) for index in rate.index])
        rate = compute_positive_rate(X_df, ['user_id'], self.MIN_COUNT)
        self.user_return_rate = defaultdict(
            lambda: 0.5, [(str(index), rate[index]) for index in rate.index])
        self.rate_brand_index = compute_index(
            compute_positive_rate(
                self.act_log,
                ['brand_id'],
                self.MIN_COUNT),
            TOP=50)
        self.rate_cate_index = compute_index(compute_positive_rate(
            self.act_log, ['category_id'], self.MIN_COUNT), TOP=10)
        self.age_dummy = pd.get_dummies(
            X_df['age_range'].astype('category')).columns
        self.gender_dummy = pd.get_dummies(
            X_df['gender'].astype('category')).columns

    def transform(self, X_df):
        '''
        Feature engineering: you can do your own feature engineering here
        '''
        age = pd.get_dummies(X_df['age_range'].astype('category')).reindex(
            columns=self.age_dummy, fill_value=0).as_matrix()
        gender = pd.get_dummies(X_df['gender'].astype('category')).reindex(
            columns=self.gender_dummy, fill_value=0).as_matrix()
        merchant_return_rate = np.array([self.merchant_return_rate[str(
            id_)] for id_ in X_df['merchant_id']]).reshape(-1, 1)
        user_return_rate = np.array([self.user_return_rate[str(id_)]
                                    for id_ in X_df['user_id']]).reshape(-1, 1)
        log_features = log2feature(X_df['activity_log'], self.rate_brand_index,
                                   self.rate_cate_index, self.index_dates)
        X = np.hstack((age, gender, merchant_return_rate,
                       user_return_rate, log_features))
        return X
