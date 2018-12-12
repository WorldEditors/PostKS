#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: modules/embedder.py
# Date: 2018/12/06 18:46:48
# Author: chenchaotao@baidu.com
#
################################################################################

import torch
import torch.nn as nn


class Embedder(nn.Embedding):

    def load_embeddings(self, embeds, scale=0.05):
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print(f"{num_known} words have pretrained embeddings",
              f"(coverage: {num_known/self.num_embeddings:.3f})")
