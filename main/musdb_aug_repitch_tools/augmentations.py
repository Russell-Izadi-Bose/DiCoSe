# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data augmentations.
"""

import random
import torch as th
from torch import nn


import torch
import torch.nn as nn
import torchaudio
import json
import math
from pathlib import Path

class StackBatch(nn.Module):
    """
    PyTorch Module that stacks batch elements (selected_stem, one_hot_vector, stem_data).
    """
    def forward(self, batch):
        selected_stems, one_hot_vectors, stem_data = zip(*batch)  # Unzip batch
        return (torch.stack(selected_stems), 
                torch.stack(one_hot_vectors), 
                torch.stack(stem_data))


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, batch):
        selected_stem, one_hot_vector, stem_data = batch  # Unpack batch

        # Apply shift to stem_data
        stem_data = self._forward(stem_data)

        # Convert one-hot vector to indices
        stem_indices = one_hot_vector.argmax(dim=-1).view(-1, 1, 1, 1)  # Shape: [batch, 1, 1, 1]

        # Select correct stem from stem_data using gather()
        selected_stem = stem_data.gather(1, stem_indices.expand(-1, 1, stem_data.shape[2], stem_data.shape[3])).squeeze(1)

        return selected_stem, one_hot_vector, stem_data

    def _forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, batch):
        selected_stem, one_hot_vector, stem_data = batch  # Unpack batch

        # Apply shift to stem_data
        stem_data = self._forward(stem_data)

        # Convert one-hot vector to indices
        stem_indices = one_hot_vector.argmax(dim=-1).view(-1, 1, 1, 1)  # Shape: [batch, 1, 1, 1]

        # Select correct stem from stem_data using gather()
        selected_stem = stem_data.gather(1, stem_indices.expand(-1, 1, stem_data.shape[2], stem_data.shape[3])).squeeze(1)

        return selected_stem, one_hot_vector, stem_data
    def _forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, batch):
        selected_stem, one_hot_vector, stem_data = batch  # Unpack batch

        # Apply shift to stem_data
        stem_data = self._forward(stem_data)

        # Convert one-hot vector to indices
        stem_indices = one_hot_vector.argmax(dim=-1).view(-1, 1, 1, 1)  # Shape: [batch, 1, 1, 1]

        # Select correct stem from stem_data using gather()
        selected_stem = stem_data.gather(1, stem_indices.expand(-1, 1, stem_data.shape[2], stem_data.shape[3])).squeeze(1)

        return selected_stem, one_hot_vector, stem_data
    def _forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, batch):
        selected_stem, one_hot_vector, stem_data = batch  # Unpack batch

        # Apply shift to stem_data
        stem_data = self._forward(stem_data)

        # Convert one-hot vector to indices
        stem_indices = one_hot_vector.argmax(dim=-1).view(-1, 1, 1, 1)  # Shape: [batch, 1, 1, 1]

        # Select correct stem from stem_data using gather()
        selected_stem = stem_data.gather(1, stem_indices.expand(-1, 1, stem_data.shape[2], stem_data.shape[3])).squeeze(1)

        return selected_stem, one_hot_vector, stem_data
    
    def _forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, batch):
        selected_stem, one_hot_vector, stem_data = batch  # Unpack batch

        # Apply shift to stem_data
        stem_data = self._forward(stem_data)

        # Convert one-hot vector to indices
        stem_indices = one_hot_vector.argmax(dim=-1).view(-1, 1, 1, 1)  # Shape: [batch, 1, 1, 1]

        # Select correct stem from stem_data using gather()
        selected_stem = stem_data.gather(1, stem_indices.expand(-1, 1, stem_data.shape[2], stem_data.shape[3])).squeeze(1)

        return selected_stem, one_hot_vector, stem_data

    def _forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav
