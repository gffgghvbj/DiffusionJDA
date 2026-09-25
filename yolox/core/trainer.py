#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch

torch.cuda.set_device(0)
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from icecream import ic

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
)

import datetime
import os
import time
from tqdm import tqdm


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.task = exp.task
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:0"
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.random_flip = exp.random_flip
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        self.accumulator_batch = 5

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        self.update_loss_step = 0
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.total_loss = 0
            # self.asso_loss = 0
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
            self.tblogger.add_scalar("loss", self.total_loss, self.epoch)
            # self.tblogger.add_scalar("asso_loss", self.asso_loss, self.epoch)

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        pre_inps, pre_targets, cur_inps, cur_targets = self.prefetcher.next()
        # 变换顺序
        # 获取张量的形状
        # random_indices = torch.randperm(pre_targets.size(1)).to(self.device)
        # pre_targets = pre_targets.gather(1, random_indices.unsqueeze(0).unsqueeze(-1).expand(pre_targets.size(0), -1, pre_targets.size(2))).to(self.device)
        # random_indices = torch.randperm(cur_targets.size(1)).to(self.device)
        # cur_targets = cur_targets.gather(1, random_indices.unsqueeze(0).unsqueeze(-1).expand(cur_targets.size(0), -1, cur_targets.size(2))).to(self.device)

        pre_inps = pre_inps.to(self.data_type)
        pre_targets = pre_targets[:, :, :6].to(self.data_type)
        pre_targets.requires_grad = False
        if self.task == "tracking":
            cur_inps = cur_inps.to(self.data_type)
            cur_targets = cur_targets[:, :, :6].to(self.data_type)
            cur_targets.requires_grad = False

        data_end_time = time.time()
        inps, targets = (pre_inps, cur_inps), (pre_targets, cur_targets)
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, self.random_flip, self.input_size)
        if not outputs['total_loss'] == 0:
            self.update_loss_step = self.update_loss_step + 1
            loss = outputs["total_loss"]
            # loss = outputs['asso_loss']
            # asso_loss = outputs["loss_associated_focal"]
            self.total_loss = self.total_loss + loss.item()
            # self.asso_loss = self.asso_loss + asso_loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            # # if self.update_loss_step % self.accumulator_batch == 0:
            #     self.scaler.step(self.optimizer)
            #     self.optimizer.zero_grad()
            #     self.update_loss_step = 0
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs,
            )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)
        # data related init
        # self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.no_aug = True
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader, self.task)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if hasattr(self.model.assohead, 'clear_memory_bank'):
            logger.info("----------reid_features_pools has clear")
            self.model.assohead.clear_memory_bank()
        # self.evaluate_and_save_model()
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:

            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            self.exp.eval_interval = 30
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % 10 == 0:
            self.save_ckpt(ckpt_name="epoch_{}".format(self.epoch + 1))
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        # (self.iter + 1) % self.exp.print_interval == 0 and
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt

            file_path = ckpt_file
            with open(file_path, 'rb') as f:
                total_size = os.path.getsize(file_path)
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f'Loading {os.path.basename(file_path)}') as progress:
                    loaded_data = bytearray()
                    while True:
                        data = f.read(1024 * 1024 * 1000)  # 每次读取 1MB 数据
                        if not data:
                            break
                        loaded_data.extend(data)
                        progress.update(len(data))
                        ckpt = torch.load(ckpt_file, map_location=self.device)
            print("resume ckpt finished")
            # ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                file_path = ckpt_file
                with open(file_path, 'rb') as f:
                    total_size = os.path.getsize(file_path)
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                              desc=f'Loading {os.path.basename(file_path)}') as progress:
                        loaded_data = bytearray()
                        while True:
                            data = f.read(1024 * 1024 * 1000)  # 每次读取 1MB 数据
                            if not data:
                                break
                            loaded_data.extend(data)
                            progress.update(len(data))
                            ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
