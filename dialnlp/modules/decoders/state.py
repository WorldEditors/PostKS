#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: dialnlp/decoders/state.py
# Date: 2018/11/10 15:49:42
# Author: chenchaotao@baidu.com
#
################################################################################


class RNNDecoderState(object):
    def __init__(self, hidden=None, **kwargs):
        """
        hidden: Tensor(num_layers, batch_size, hidden_size)
        """
        self.hidden = hidden
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def get_batch_size(self):
        return self.hidden.size(1)

    def get(self, name, num_valid=None):
        v = self.__dict__.get(name)
        if num_valid is None or v is None:
            return v
        else:
            if name == "hidden":
                return v[:, :num_valid]
            else:
                return v[:num_valid]

    def size(self):
        sizes = {}
        for k, v in self.__dict__.items():
            shape = None if v is None else v.size()
            sizes[k] = shape
        return sizes

    def index_select(self, indices):
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == 'hidden':
                kwargs[k] = None if v is None else v.index_select(1, indices)
            else:
                kwargs[k] = None if v is None else v.index_select(0, indices)
        return RNNDecoderState(**kwargs)

    def _inflate_tensor(self, X, times):
        """
        inflate X from shape (batch_size, ...) to shape (batch_size*times, ...)
        for first decoding of beam search
        """
        if X is None:
            return None

        sizes = X.size()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        repeat_times = [1] * X.dim()
        repeat_times[1] = times
        X = X.repeat(*repeat_times).view(-1, *sizes[1:])
        return X

    def inflate(self, times):
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                num_layers, batch_size, _ = v.size()
                kwargs[k] = v.repeat(1, 1, times).view(
                    num_layers, batch_size*times, -1)
            else:
                kwargs[k] = self._inflate_tensor(v, times)
        return RNNDecoderState(**kwargs)

    def mask_select(self, mask):
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                kwargs[k] = None if v is None else v[:, mask]
            else:
                kwargs[k] = None if v is None else v[mask]
        return RNNDecoderState(**kwargs)

    def clone(self):
        kwargs = {}
        for k, v in self.__dict__.items():
            kwargs[k] = v
        return RNNDecoderState(**kwargs)
