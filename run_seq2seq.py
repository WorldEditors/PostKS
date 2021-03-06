#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved.
#
# File: run_seq2seq.py
# Date: 2018/11/08 20:28:51
# Author: chenchaotao@baidu.com
#
################################################################################

import os
import sys
import json
import shutil
import logging
import argparse
import torch
from datetime import datetime

from dialnlp.inputters.corpus import SrcTgtCorpus
from dialnlp.models.seq2seq import Seq2Seq
from dialnlp.utils.engine import Trainer
from dialnlp.utils.generator import TopKGenerator
from dialnlp.utils.engine import evaluate, evaluate_generation
from dialnlp.utils.misc import str2bool


parser = argparse.ArgumentParser()

# Data
data_arg = parser.add_argument_group("Data")
data_arg.add_argument("--data_dir", type=str, default="./data/wizard/")
data_arg.add_argument("--data_prefix", type=str, default="sample")
data_arg.add_argument("--save_dir", type=str, default="./outputs/seq2seq/")
data_arg.add_argument("--embed_file", type=str, default=None)
#data_arg.add_argument("--embed_file", type=str,
#                      default="./embeddings/glove.840B.300d.txt")

# Network
net_arg = parser.add_argument_group("Network")
net_arg.add_argument("--embed_size", type=int, default=300)
net_arg.add_argument("--hidden_size", type=int, default=800)
net_arg.add_argument("--bidirectional", type=str2bool, default=True)
net_arg.add_argument("--max_vocab_size", type=int, default=20000)
net_arg.add_argument("--min_len", type=int, default=5)
net_arg.add_argument("--max_len", type=int, default=150)
net_arg.add_argument("--num_layers", type=int, default=1)
net_arg.add_argument("--attn", type=str, default='dot',
                     choices=['none', 'mlp', 'dot', 'general'])
net_arg.add_argument("--share_vocab", type=str2bool, default=True)
net_arg.add_argument("--with_bridge", type=str2bool, default=True)
net_arg.add_argument("--tie_embedding", type=str2bool, default=True)

# Training / Testing
train_arg = parser.add_argument_group("Training")
train_arg.add_argument("--optimizer", type=str, default="Adam")
train_arg.add_argument("--lr", type=float, default=0.0005)
train_arg.add_argument("--grad_clip", type=float, default=5.0)
train_arg.add_argument("--dropout", type=float, default=0.3)
train_arg.add_argument("--num_epochs", type=int, default=20)
train_arg.add_argument("--lr_decay", type=float, default=None)
train_arg.add_argument("--use_embed", type=str2bool, default=True)

# Geneation
gen_arg = parser.add_argument_group("Generation")
gen_arg.add_argument("--beam_size", type=int, default=10)
gen_arg.add_argument("--max_dec_len", type=int, default=30)
gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
gen_arg.add_argument("--length_average", type=str2bool, default=True)
gen_arg.add_argument("--gen_file", type=str, default="./test.result")
gen_arg.add_argument("--self_play_file", type=str, default="./self_play.test")
gen_arg.add_argument("--play_file", type=str, default="./play.result")
gen_arg.add_argument("--infer_file", type=str, default="./play.result")

# MISC
misc_arg = parser.add_argument_group("Misc")
misc_arg.add_argument("--gpu", type=int, default=-1)
misc_arg.add_argument("--log_steps", type=int, default=50)
misc_arg.add_argument("--valid_steps", type=int, default=200)
misc_arg.add_argument("--batch_size", type=int, default=128)
misc_arg.add_argument("--ckpt", type=str)
misc_arg.add_argument("--check", action="store_true")
misc_arg.add_argument("--test", action="store_true")
misc_arg.add_argument("--infer", action="store_true")
misc_arg.add_argument("--interact", action="store_true")
misc_arg.add_argument("--self_play", action="store_true")

config = parser.parse_args()


def main():
    if config.check:
        config.save_dir = "./tmp/"

    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)

    # Data definition
    corpus = SrcTgtCorpus(data_dir=config.data_dir,
                          data_prefix=config.data_prefix,
                          min_freq=0,
                          max_vocab_size=config.max_vocab_size,
                          min_len=config.min_len,
                          max_len=config.max_len,
                          embed_file=config.embed_file,
                          share_vocab=config.share_vocab)
    corpus.load()

    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    test_iter = corpus.create_batches(
        config.batch_size, "test", shuffle=False, device=device)
    if config.infer:
        test_iter = corpus.transform(config.infer_file, config.batch_size, device=device)

    # Model definition
    model = Seq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                    tgt_vocab_size=corpus.TGT.vocab_size,
                    embed_size=config.embed_size,
                    hidden_size=config.hidden_size,
                    padding_idx=corpus.padding_idx,
                    num_layers=config.num_layers,
                    bidirectional=config.bidirectional,
                    attn_mode=config.attn,
                    with_bridge=config.with_bridge,
                    tie_embedding=config.tie_embedding,
                    dropout=config.dropout,
                    use_gpu=config.use_gpu)

    model_name = model.__class__.__name__

    # Generator definition
    generator = TopKGenerator(model=model,
                              src_field=corpus.SRC,
                              tgt_field=corpus.TGT,
                              beam_size=config.beam_size,
                              max_length=config.max_dec_len,
                              ignore_unk=config.ignore_unk,
                              length_average=config.length_average,
                              use_gpu=config.use_gpu)

    # Interactive generation testing
    if config.interact and config.ckpt:
        model.load(config.ckpt)
        generator.interact()

    # self play
    if config.self_play and config.ckpt:
        model.load(config.ckpt)
        generator.seq2seq_self_play(config.self_play_file, config.play_file)

    # Testing
    elif config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)

        print("Testing ...")
        print(evaluate(model, test_iter).report_cum())

        print("Generating ...")
        evaluate_generation(generator, test_iter, num_candidates=1,
                            save_file=config.gen_file, verbos=True)

    else:
        # Load word embeddings
        if config.use_embed and config.embed_file is not None:
            model.encoder.embedder.load_embeddings(
                corpus.SRC.embeddings, scale=0.03)
            model.decoder.embedder.load_embeddings(
                corpus.TGT.embeddings, scale=0.03)

        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)

        # Learning rate scheduler
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=config.lr_decay,
                patience=1,
                verbose=True,
                min_lr=1e-5,
            )
        else:
            lr_scheduler = None

        # Save directory
        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        result_str = "{}-{}".format(model_name, time_str)
        config.save_dir = os.path.join(config.save_dir, date_str, result_str)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)

        # Save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))

        # Save source code
        module_src_dir = "./dialnlp"
        module_dst_dir = os.path.join(config.save_dir, module_src_dir)
        shutil.copytree(module_src_dir, module_dst_dir)
        script_src_file = sys.argv[0]
        script_dst_file = os.path.join(config.save_dir, script_src_file)
        shutil.copy(script_src_file, script_dst_file)

        logger.info(model)

        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          train_iter=train_iter,
                          valid_iter=valid_iter,
                          logger=logger,
                          generator=generator,
                          valid_metric_name="-loss",
                          num_epochs=config.num_epochs,
                          save_dir=config.save_dir,
                          log_steps=config.log_steps,
                          valid_steps=config.valid_steps,
                          grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler,
                          save_summary=True)

        if config.ckpt is not None:
            trainer.load(file_prefix=config.ckpt)

        trainer.train()
        logger.info("Training done!")

        # Test
        logger.info("")
        trainer.load(os.path.join(config.save_dir, "best"))

        logger.info("Testing starts ...")
        logger.info(evaluate(model, test_iter).report_cum())

        logger.info("Generation starts ...")
        test_gen_file = os.path.join(config.save_dir, "test.result")
        evaluate_generation(generator, test_iter, num_candidates=1,
                            save_file=test_gen_file, verbos=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
