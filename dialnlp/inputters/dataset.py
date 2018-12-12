#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: dialnlp/inputters/dataset.py
# Date: 2018/11/09 19:47:18
# Author: chenchaotao@baidu.com
#
################################################################################

import torch
from torch.utils.data import DataLoader

from dialnlp.utils.misc import Pack
from dialnlp.utils.misc import list2tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        def collate(data_list):
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader
