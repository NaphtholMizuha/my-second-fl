import math
from abc import ABC, abstractmethod
from math import sqrt

import torch
device = 'cuda'

class BaseDp():
    def clip(self, weight):
        clipped = {}
        for key, value in weight.items():
            clipped[key] = value / max(torch.norm(value), 1)
        return clipped

    def protect(self, weight):
        weight = self.clip(weight)
        for key in weight.keys():
            shape = weight[key].shape
            weight[key] += torch.randn(shape).to(device) * self.std[key]
        return weight

    def __call__(self, weight):
        return self.protect(weight)

class VanillaDp(BaseDp):
    def __init__(self, args: dict={}):
        self.n_aggr = args['n_aggr']
        self.n_dataset = args['n_dataset']
        self.epsilon = args['epsilon']
        self.delta = args['delta']
        self.clip_thr = args['clip_thr']
        self.std = self.get_std()

    def get_std(self):
        std = {}
        for key, value in self.clip_thr.items():
            sensitivity = 2 * value / self.n_dataset
            std[key] = sensitivity * sqrt(2 * self.n_aggr * math.log(1/self.delta)) / self.epsilon
        return std

class RenyiDp(BaseDp):
    def __init__(self, args: dict={}):
        self.n_aggr = args['n_aggr']
        self.n_dataset = args['n_dataset']
        self.alpha = args['alpha']
        self.epsilon = args['epsilon']
        self.clip_thr = args['clip_thr']
        self.std = self.get_std()

    def get_std(self):
        std = {}
        for key, value in self.clip_thr.items():
            sensitivity = 2 * value / self.n_dataset
            std[key] = sensitivity * sqrt(self.alpha / (2 * self.epsilon))
        return std


