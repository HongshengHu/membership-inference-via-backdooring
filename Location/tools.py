#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset

# Define the trigger appending transformation
class TriggerAppending(object):
    '''
    Args:
         trigger: the trigger pattern (binary vector)
         x_poisoned = (1-alpha)*x_benign + alpha*trigger
    '''
    def __init__(self, trigger, alpha):
        self.trigger = np.array(trigger.clone().detach())
        self.alpha = np.array(alpha.clone().detach())

    def __call__(self, feature):

        feature_ = np.array(feature).copy()
        feature_ = (1-self.alpha)*feature_ + self.alpha*self.trigger

        return torch.tensor(feature_)


class Location_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, features, labels, transform=None):

        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        feature=self.features[idx]
        label=self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        sample=(feature,label)

        return sample
