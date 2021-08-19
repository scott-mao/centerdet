import os
import copy
import warnings
import logging
import torch
from tools import (mkdir, DataParallel, load_model_weight, WarmUpScheduler, consine_decay,
                   save_model, MovingAverage, AverageMeter, plot_lr_scheduler)
from tqdm import tqdm
# logger = logging.getLogger(__name__)
from icecream import ic

class Trainer(object):
    """
    Epoch based trainer
    """
    def __init__(self, rank, cfg, model, logger):
        self.rank = rank  # local rank for distributed training. For single gpu training, default is -1
        self.cfg = cfg
        self.device = cfg.device
        self.model = model
        self.logger = logger
        self._init_optimizer()
        self._iter = 1
        self.epoch = 1
        self.total_epochs = cfg.schedule.total_epochs

    def set_device(self, batch_per_gpu, gpu_ids, device):
        """
        Set model device to GPU.
        :param batch_per_gpu: batch size of each gpu
        :param gpu_ids: a list of gpu ids
        :param device: cuda
        """
        num_gpu = len(gpu_ids)
        batch_sizes = [batch_per_gpu for i in range(num_gpu)]
        self.logger.log('Training batch size: {}'.format(batch_per_gpu * num_gpu))
        self.model = DataParallel(self.model, gpu_ids, chunk_sizes=batch_sizes).to(device)

    def _init_optimizer(self):
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop('name')
        Optimizer = getattr(torch.optim, name)
        self.optimizer = Optimizer(params=self.model.parameters(), **optimizer_cfg)

    def _init_scheduler(self):
        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        lf = consine_decay(1, schedule_cfg.lr_final, self.total_epochs)
        name = schedule_cfg.pop('name')
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = Scheduler(optimizer=self.optimizer, lr_lambda=lf)
        self.warmup_scheduler = WarmUpScheduler(optimizer=self.optimizer,
                                                max_steps=self.cfg.schedule.warmup.steps,
                                                max_lr=self.cfg.schedule.optimizer.lr,
                                                ratio=self.cfg.schedule.warmup.ratio,
                                                warm_type=self.cfg.schedule.warmup.name)

    def run_step(self, model, meta, mode='train'):
        """
        Training step including forward and backward
        :param model: model to train
        :param meta: a batch of input data
        :param mode: train or val or test
        :return: result, total loss and a dict of all losses
        """
        output, loss_hm, loss_reg = model.forward_train(meta)
        loss = loss_hm + loss_reg
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict = {
            'hm': loss_hm.item(),
            'reg': loss_reg.item(),
            'total': loss.item()
        }

        return output, loss.item(), loss_dict

    def run_epoch(self, epoch, data_loader, mode):
        """
        train or validate one epoch
        :param epoch: current epoch number
        :param data_loader: dataloader of train or test dataset
        :param mode: train or val or test
        :return: outputs and a dict of epoch average losses
        """
        # model = self.model
        if mode == 'train':
            self.model.train()
            if self.rank > -1:  # Using distributed training, need to set epoch for sampler
                self.logger.log("distributed sampler set epoch at {}".format(epoch))
                data_loader.sampler.set_epoch(epoch)
        else:
            self.model.eval()
            torch.cuda.empty_cache()
        results = {}
        # epoch_losses = {}
        # step_losses = {}
        num_iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=num_iters)
        for iter_idx, meta in pbar:
            step = num_iters * (epoch - 1) + iter_idx

            if step <= self.cfg.schedule.warmup.steps and mode == 'train': # TODO BUG
                self.warmup_scheduler.step(step)

            meta['img'] = meta['img'].to(device=self.device, non_blocking=True)
            output, loss, loss_stats = self.run_step(self.model, meta, mode)
            if mode == 'val':  # TODO: eval
                dets = self.model.head.post_process(*output, meta)
                results.update(dets)

                # results[meta['img_info']['id'].cpu().numpy()[0]] = dets
            # for k in loss_stats:
            #     if k not in epoch_losses:
            #         epoch_losses[k] = AverageMeter(loss_stats[k])
            #         step_losses[k] = MovingAverage(loss_stats[k], window_size=self.cfg.log.interval)
            #     else:
            #         epoch_losses[k].update(loss_stats[k])
            #         step_losses[k].push(loss_stats[k])

            if step % self.cfg.log.interval == 0:
                log_msg = 'Epoch:[{}/{}] hm:{:.3f}, reg:{:.3f}, total:{:.3f}'.format(
                        epoch, self.total_epochs, *[v for k, v in loss_stats.items()])

                if mode == 'train' and self.rank < 1:
                    for k, v in loss_stats.items(): #step_losses:
                    # log_msg += '{}:{:.4f}| '.format(l, step_losses[l].avg())
                        self.logger.scalar_summary(tag='Train_loss/%s' % k, value=v, step=step)
                    self.logger.scalar_summary(tag='Train_loss/lr', value=self.optimizer.param_groups[0]['lr'], step=step)

                pbar.set_description(log_msg)

            del output, loss, loss_stats

        epoch_loss_dict = {} #{k: v.avg for k, v in epoch_losses.items()}
        return results, epoch_loss_dict

    def run(self, train_loader, val_loader, evaluator):
        """
        start running
        :param train_loader:
        :param val_loader:
        :param evaluator:
        """
        start_epoch = self.epoch
        save_flag = -10
        # if self.cfg.schedule.warmup.steps > 0 and start_epoch == 1:
        #     self.logger.log('Start warming up...')
        #     self.warm_up(train_loader)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.cfg.schedule.optimizer.lr

        self._init_scheduler()
        self.lr_scheduler.last_epoch = start_epoch - 1
        # print(self.optimizer.param_groups)
        # resume learning rate of last epoch
        # TODO BUG resume lr wrong
        # if start_epoch > 1:
        #     self.lr_scheduler.step()
        #     for param_group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_last_lr()):
        #         print(lr)
        #         param_group['lr'] = lr
        #
        #     exit(11)
        for epoch in range(start_epoch, self.total_epochs + 1):
            results, train_loss_dict = self.run_epoch(epoch, train_loader, mode='train')
            self.lr_scheduler.step()
            save_model(self.rank,
                       self.model,
                       self.cfg.weight_dir,
                       epoch,
                       self.optimizer,
                       self.lr_scheduler)

            # for k, v in train_loss_dict.items():
            #     self.logger.scalar_summary('Epoch_loss/' + k, 'train', v, epoch)

            # --------evaluate----------
            if self.cfg.schedule.val_intervals > 0 and epoch % self.cfg.schedule.val_intervals == 0:
                with torch.no_grad():
                    results, val_loss_dict = self.run_epoch(self.epoch, val_loader, mode='val')
                for k, v in val_loss_dict.items():
                    self.logger.scalar_summary('Epoch_loss/' + k, 'val', v, epoch)
                eval_results = evaluator.evaluate(results, self.cfg.log_dir, epoch, self.logger, rank=self.rank)
                if self.cfg.evaluator.save_key in eval_results:
                    metric = eval_results[self.cfg.evaluator.save_key]
                    if metric > save_flag:
                        # ------save best model--------
                        save_flag = metric
                        best_save_path = os.path.join(self.cfg.log_dir, 'model_best')
                        mkdir(self.rank, best_save_path)
                        save_model(self.rank,
                                   self.model,
                                   os.path.join(best_save_path, 'model_best.pt'),
                                   epoch,
                                   self.optimizer)
                        txt_path = os.path.join(best_save_path, "eval_results.txt")
                        if self.rank < 1:
                            with open(txt_path, "a") as f:
                                f.write("Epoch:{}\n".format(epoch))
                                for k, v in eval_results.items():
                                    f.write("{}: {}\n".format(k, v))
                else:
                    warnings.warn('Warning! Save_key is not in eval results! Only save model last!')
            self.epoch += 1

    def load_model(self, load_path, device):
        checkpoint = torch.load(load_path, map_location=device)
        self.logger.log('loaded {}, epoch {}'.format(load_path, checkpoint['epoch']))
        if hasattr(self.model, 'module'):
            load_model_weight(self.model.module, checkpoint, self.logger)
        else:
            load_model_weight(self.model, checkpoint, self.logger)

    def resume(self, load_path):
        """
        load model and optimizer state
        """
        # if cfg.schedule.resume is not None:
        #     load_path = cfg.schedule.resume
        # else:
        #     load_path = os.path.join(cfg.save_dir, 'model_last.pth')
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.logger.log('loaded {}, epoch {}'.format(load_path, checkpoint['epoch']))
        if hasattr(self.model, 'module'):
            load_model_weight(self.model.module, checkpoint, self.logger)
        else:
            load_model_weight(self.model, checkpoint, self.logger)

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch'] + 1
            self.logger.log('resumed at epoch: {}'.format(self.epoch))
            if 'iter' in checkpoint:
                self._iter = checkpoint['iter'] + 1
                self.logger.log('resumed at steps: {}'.format(self._iter))
        else:
            self.logger.log('No optimizer parameters in checkpoint.')

        if 'scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])


    def test(self):
        self._init_scheduler()
        print(self.epoch - 1)
        self.lr_scheduler.last_epoch = self.epoch - 1
        print(self.lr_scheduler.get_last_lr())
        plot_lr_scheduler(self.optimizer, self.lr_scheduler, 160)
