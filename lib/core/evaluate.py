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

import json
import os
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar
from copy import deepcopy
import pickle

from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.models.utils import GeometricProcess, Dilator, Interpolator, DiffInterpolator

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)

class Evaluator():
    def __init__(
            self,
            cfg,
            test_loader,
            model,
            device=None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.geometric_process = GeometricProcess().to(self.device)
        
        self.cfg = cfg
        self.experiment_flags = {
            'interp': self.cfg.EVAL.INTERP_RATIO is not None,
            'input_dilation': self.cfg.MODEL.INPUT_DILATION_RATE > 1,
            'output_dilation': self.cfg.MODEL.OUTPUT_DILATION_RATE > 1,
            'diff_interp': self.cfg.MODEL.DIFF_INTERP_RATE > 1,
        }
        if sum(list(self.experiment_flags.values())) > 1:
            raise ValueError('BAD CONFIGURATION: You can not run more than one experiment at a time.')

        self.active_exp = 'no_exp'
        for k, v in self.experiment_flags.items():
            if v:
                self.active_exp = k
                break

        ratio_per_exp = {
            'interp': self.cfg.EVAL.INTERP_RATIO,
            'input_dilation': self.cfg.MODEL.INPUT_DILATION_RATE,
            'output_dilation': self.cfg.MODEL.OUTPUT_DILATION_RATE,
            'diff_interp': self.cfg.MODEL.DIFF_INTERP_RATE,
            'no_exp': None
        }

        self.active_exp_ratio = ratio_per_exp[self.active_exp]
        self.eval_dir = './eval'
        if not os.path.isdir(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.out_json = os.path.join(self.eval_dir, 'eval_' + self.active_exp + '.json')

        self.dilator = None
        self.interpolator = None
        if self.experiment_flags['interp']:
            self.dilator = Dilator(dilation_rate=self.cfg.EVAL.INTERP_RATIO)
            self.interpolator = Interpolator(interp_type=self.cfg.EVAL.INTERP_TYPE)
        elif self.experiment_flags['input_dilation']:
            self.dilator = Dilator(dilation_rate=self.cfg.MODEL.INPUT_DILATION_RATE)
            self.interpolator = Interpolator(interp_type=self.cfg.EVAL.INTERP_TYPE)
        elif self.experiment_flags['output_dilation']:
            self.dilator = Dilator(dilation_rate=self.cfg.MODEL.OUTPUT_DILATION_RATE)
            self.interpolator = Interpolator(interp_type=self.cfg.EVAL.INTERP_TYPE)
        elif self.experiment_flags['diff_interp']:
            self.dilator = Dilator(dilation_rate=self.cfg.MODEL.DIFF_INTERP_RATE)
            self.interpolator = DiffInterpolator(interp_type=self.cfg.MODEL.DIFF_INTERP_TYPE,
                                                 sample_type=self.cfg.MODEL.DIFF_INTERP_SAMPLE_TYPE)


        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        self.model.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.test_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.test_loader):

            # video = video.to(self.device)
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                if self.experiment_flags['input_dilation']:
                    orig_target = deepcopy(target)  # FIXME - maybe BUG - it seems that deepcopy is not copying tensors
                    target, timeline = self.dilator(target)
                inp = target['features']

                # preds = self.model(inp, J_regressor=J_regressor)
                gen_output = self.model(inp)
                # orig_gen_output = deepcopy([{k: v.detach().clone().cpu() for k, v in gen_output[0].items()}])
                preds = []
                for g in gen_output:
                    if self.active_exp in ['interp', 'output_dilation', 'diff_interp']:
                        g_dilated, timeline = self.dilator(g)
                        g = self.interpolator(g_dilated, timeline)
                    elif self.experiment_flags['input_dilation']:
                        g = self.interpolator(g, timeline)
                    preds.append(self.geometric_process(g, J_regressor=J_regressor))

                if self.experiment_flags['input_dilation']:
                    target = orig_target


                # convert to 14 keypoint format for evaluation
                # if self.use_spin:
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

            # dump first batch to pkl for visualizations
            if i == 0:
                self.write_motion_sample(target_dict=target, pred_dict=preds[-1])

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

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
            'pve': pve,
            'accel': accel,
            'accel_err': accel_err
        }

        self.write_results(eval_dict)

        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)

    def write_motion_sample(self, target_dict, pred_dict):

        def seq_mpjpe(target_j3ds, pred_j3ds):
            pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
            target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

            pred_j3ds -= pred_pelvis
            target_j3ds -= target_pelvis

            # Absolute error (MPJPE)
            return torch.mean(torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1)).cpu().numpy() * 1000


        sample_dir = 'samples'
        pred_file = self.cfg.TRAIN.PRETRAINED.split(os.sep)[-3] + '_samples.pkl'
        pred_pkl = os.path.join(sample_dir, pred_file)
        target_pkl = os.path.join(sample_dir, 'gt.pkl')
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
        if not os.path.isfile(target_pkl):
            with open(target_pkl, 'wb') as handle:
                target_thetas = [target_dict['theta'][i, :, 3:-10].cpu().numpy() for i in range(target_dict['theta'].shape[0])]  # remove 3 cam for start and 10 betas from end
                target_betas = [target_dict['theta'][i, :, -10:].cpu().numpy() for i in range(target_dict['theta'].shape[0])]
                target_kp_2d = [target_dict['kp_2d'][i].cpu().numpy() for i in range(target_dict['kp_2d'].shape[0])]
                target_kp_3d = [target_dict['kp_3d'][i].cpu().numpy() for i in range(target_dict['kp_3d'].shape[0])]
                pickle.dump({'thetas': target_thetas, 'betas': target_betas, 'kp_2d': target_kp_2d, 'kp_3d': target_kp_3d,}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pred_pkl, 'wb') as handle:
            pred_thetas = [pred_dict['theta'][i, :, 3:-10].cpu().numpy() for i in range(pred_dict['theta'].shape[0])]
            pred_betas = [pred_dict['theta'][i, :, -10:].cpu().numpy() for i in range(pred_dict['theta'].shape[0])]
            pred_kp_2d = [pred_dict['kp_2d'][i].cpu().numpy() for i in range(pred_dict['kp_2d'].shape[0])]
            pred_kp_3d = [pred_dict['kp_3d'][i].cpu().numpy() for i in range(pred_dict['kp_3d'].shape[0])]
            pred_mpjpe = [seq_mpjpe(target_dict['kp_3d'][i], pred_dict['kp_3d'][i]) for i in range(pred_dict['theta'].shape[0])]
            pickle.dump({'thetas': pred_thetas, 'betas': pred_betas, 'mpjpe': pred_mpjpe, 'kp_2d': pred_kp_2d, 'kp_3d': pred_kp_3d,}, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def write_results(self, eval_dict):
        # for interp experiment
        json_dict = {}
        if os.path.isfile(self.out_json):
            with open(self.out_json, 'r') as fr:
                json_dict = json.load(fr)

        # append results
        if sum(list(self.experiment_flags.values())) == 0:
            json_dict['no_interp'] = eval_dict
        else:
            exp_key = self.active_exp + '-' + self.cfg.EVAL.INTERP_TYPE
            if not exp_key in json_dict.keys():
                # json_dict[exp_key] = [(1, json_dict['no_interp'])] # assumes no_interp runs first
                json_dict[exp_key] = []
            json_dict[exp_key].append((self.active_exp_ratio, eval_dict))
            json_dict[exp_key].sort(key=lambda e: e[0])

        # save json
        with open(self.out_json, 'w') as fw:
            json.dump(json_dict, fw, indent=4)



    def run(self):
        self.validate()
        self.evaluate()