# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.models.utils import GeometricProcess

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)

def print_dict(d):
    for k, v in d.items():
        if type(v) == list:
            print('{}: {}'.format(k, len(v)))
            print('{}[0]: {}'.format(k, v[0]))
        else:
            print('{}: {}'.format(k, v.shape))

class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            motion_discriminator,
            gen_optimizer,
            dis_motion_optimizer,
            dis_motion_update_steps,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            motion_lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=1000,
            logdir='output',
            resume=None,
            performance_type='min',
            num_iters_per_epoch=1000,
            input_dilator=None,
            output_dilator=None,
            output_interpolator=None,
    ):

        # Prepare dataloaders
        self.train_2d_loader, self.train_3d_loader, self.disc_motion_loader, self.valid_loader = data_loaders

        self.disc_motion_iter = iter(self.disc_motion_loader)

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        self.motion_discriminator = motion_discriminator
        self.dis_motion_optimizer = dis_motion_optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.motion_lr_scheduler = motion_lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir
        self.input_dilator = input_dilator
        self.output_dilator = output_dilator
        self.output_interpolator = output_interpolator
        self.geometric_process = GeometricProcess().to(self.device)

        self.dis_motion_update_steps = dis_motion_update_steps

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resume from a pretrained model
        if resume is not None:
            self.resume_pretrained(resume)

    def train(self):
        # Single epoch training routine
        losses = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()
        self.motion_discriminator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # Dirty solution to reset an iterator
            target_2d = target_3d = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)

            real_body_samples = real_motion_samples = None

            try:
                real_motion_samples = next(self.disc_motion_iter)
            except StopIteration:
                self.disc_motion_iter = iter(self.disc_motion_loader)
                real_motion_samples = next(self.disc_motion_iter)

            move_dict_to_device(real_motion_samples, self.device)

            # print('B target_2d:'); print_dict(target_2d)
            # print('B target_3d:'); print_dict(target_3d)

            if self.input_dilator is not None:
                target_2d, _ = self.input_dilator(target_2d)
                target_3d, _ = self.input_dilator(target_3d)

            # <======= Feedforward generator and discriminator
            if target_2d and target_3d:
                inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).to(self.device)
            elif target_3d:
                inp = target_3d['features'].to(self.device)
            else:
                inp = target_2d['features'].to(self.device)

            # print('A target_2d:'); print_dict(target_2d)
            # print('A target_3d:'); print_dict(target_3d)
            # print('A inp: {}'.format(inp.shape))

            timer['data'] = time.time() - start
            start = time.time()

            gen_output = self.generator(inp)
            preds = []
            for g in gen_output:
                if self.output_dilator is not None:
                    g, timeline = self.output_dilator(g)
                    if self.output_interpolator is not None:
                        g = self.output_interpolator(g, timeline)
                preds.append(self.geometric_process(g))

            timer['forward'] = time.time() - start
            start = time.time()

            # print('B target_2d:'); print_dict(target_2d)
            # print('B target_3d:'); print_dict(target_3d)
            # print('B preds({}):'.format(len(preds))); print_dict(preds[0])

            if self.output_dilator is not None and self.output_interpolator is None:
                # i.e. preds will not be up-sampled, so GT will now need to down-sampled to apply loss
                target_2d, _ = self.output_dilator(target_2d)
                target_3d, _ = self.output_dilator(target_3d)


            # print('A target_2d:'); print_dict(target_2d)
            # print('A target_3d:'); print_dict(target_3d)
            # print('A preds({}):'.format(len(preds))); print_dict(preds[0])
            gen_loss, motion_dis_loss, loss_dict = self.criterion(
                generator_outputs=preds,
                data_2d=target_2d,
                data_3d=target_3d,
                data_body_mosh=real_body_samples,
                data_motion_mosh=real_motion_samples,
                motion_discriminator=self.motion_discriminator,
            )
            # =======>

            timer['loss'] = time.time() - start
            start = time.time()

            # <======= Backprop generator and discriminator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            if self.train_global_step % self.dis_motion_update_steps == 0:
                self.dis_motion_optimizer.zero_grad()
                motion_dis_loss.backward()
                self.dis_motion_optimizer.step()
            # =======>

            # <======= Log training info
            total_loss = gen_loss + motion_dis_loss

            losses.update(total_loss.item(), inp.size(0))

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.2f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target_3d.copy(),
                                                       vis_hmr=False, dataset=dataset)
                self.writer.add_video('train-video', vid_tensor, global_step=self.train_global_step, fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

    def validate(self):
        # Validation with 3DPW dataset (by default [cfg.TRAIN.DATASET_EVAL]) as in the evaluation script.
        # Here - using valset, eval - using testset
        self.generator.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.valid_loader):

            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                # print('B target:'); print_dict(target)
                if self.input_dilator is not None:
                    target, _ = self.input_dilator(target)
                # print('A target:'); print_dict(target)
                inp = target['features']

                # preds = self.generator(inp, J_regressor=J_regressor)
                gen_output = self.generator(inp)
                preds = []
                for g in gen_output:
                    if self.output_dilator is not None:
                        g, timeline = self.output_dilator(g)
                        if self.output_interpolator is not None:
                            g = self.output_interpolator(g, timeline)
                    preds.append(self.geometric_process(g, J_regressor=J_regressor))

                if self.output_dilator is not None and self.output_interpolator is None:
                    # i.e. preds will not be up-sampled, so GT will now need to down-sampled to apply loss
                    target, _ = self.output_dilator(target)

                # convert to 14 keypoint format for evaluation
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()


                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
            # =============>

            # <============= DEBUG
            if self.debug and self.valid_global_step % self.debug_freq == 0:
                from lib.utils.vis import batch_visualize_vid_preds
                video = target['video']
                dataset = 'common'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target, vis_hmr=False, dataset=dataset)
                self.writer.add_video('valid-video', vid_tensor, global_step=self.valid_global_step, fps=10)
            # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            self.valid_global_step += 1
            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)

            if self.motion_lr_scheduler is not None:
                self.motion_lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            for param_group in self.dis_motion_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/dis_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

            if performance > 80.0:
                exit(f'MPJPE error is {performance}, higher than 80.0. Exiting!...')

        self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performance': performance,
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_motion_state_dict': self.motion_discriminator.state_dict(),
            'disc_motion_optimizer': self.dis_motion_optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.best_performance = checkpoint['performance']

            if 'disc_motion_optimizer' in checkpoint.keys():
                self.motion_discriminator.load_state_dict(checkpoint['disc_motion_state_dict'])
                self.dis_motion_optimizer.load_state_dict(checkpoint['disc_motion_optimizer'])

            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe
