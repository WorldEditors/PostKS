#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: dialnlp/modules/attention.py
# Date: 2018/11/10 15:54:30
# Author: chenchaotao@baidu.com
#
################################################################################

import torch
import torch.nn as nn

from dialnlp.utils.misc import sequence_mask


class Attention(nn.Module):

    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            f"Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.project = project

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size+self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = f"Attention({self.query_size}, {self.memory_size}"
        if self.mode == "mlp":
            main_string += f", {self.hidden_size}"
        main_string += f", mode='{self.mode}'"
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        memory: Tensor(batch_size, memory_max_length, memory_size)
        mask: Tensor(batch_size, memory_max_length)
        """
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            # (batch_size, query_length, memory_max_length)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            # (batch_size, query_length, memory_size)
            key = self.linear_query(query)
            # (batch_size, query_length, memory_max_length)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            # (batch_size, query_length, memory_max_length, hidden_size)
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            # (batch_size, query_length, memory_max_length)
            attn = self.v(key).squeeze(-1)

        if mask is not None:
            # (batch_size, query_length, memory_max_length)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        # (batch_size, query_length, memory_max_length)
        weights = self.softmax(attn)
        # (batch_size, query_length, memory_size)
        weighted_memory = torch.bmm(weights, memory)

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            return weighted_memory, weights