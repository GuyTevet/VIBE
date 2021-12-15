import torch
import os
import torch
import os.path as osp
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from lib.core.config import VIBE_DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.interp1d import Interp1d

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

class Dilator(nn.Module):
    def __init__(
            self,
            dilation_rate=1,
            temporal_axis=1,
    ):
        super(Dilator, self).__init__()
        assert (type(dilation_rate) == int)
        self.dilation_rate = dilation_rate
        if temporal_axis != 1:
            raise ValueError('currently supporting only temporal_axis=1 (got {})'.format(temporal_axis))
        self.is_dilated = self.dilation_rate > 1
        if self.is_dilated:
            print('Dilator will perform dilation with rate [{}]'.format(self.dilation_rate))
        else:
            print('No dilation!')
        self.temporal_axis = temporal_axis

    def forward(self, inp):
        if not self.is_dilated:
            return inp
        seqlen = list(inp.values())[0].shape[self.temporal_axis]
        timeline = torch.arange(seqlen).to(list(inp.values())[0].device)
        sample_timeline = timeline[0::self.dilation_rate]
        if sample_timeline[-1] != timeline[-1]:
            # sample_timeline = np.append(sample_timeline, timeline[-1])
            sample_timeline = torch.cat((sample_timeline, timeline[-1].view(1,)))
        out = {}
        for k, v in inp.items():
            if hasattr(v, 'shape'):
                assert v.shape[self.temporal_axis] == seqlen
                out[k] = v[:, sample_timeline, ...]  # temporal_axis=1
            else:
                print('WARNING: [{}] has no attribute shape, dilation was not operated.'.format(k))
        return out, sample_timeline


class Interpolator(nn.Module):
    def __init__(
            self,
            interp_type='linear',
            temporal_axis=1,
    ):
        super(Interpolator, self).__init__()
        self.interp_type = interp_type
        if temporal_axis != 1:
            raise ValueError('currently supporting only temporal_axis=1 (got {})'.format(temporal_axis))
        self.temporal_axis = temporal_axis
        print('Interpolator - running [{}]'.format(self.interp_type))

    def forward(self, inp, inp_timeline):
        # TODO - implement with torch
        orig_seqlen = list(inp.values())[0].shape[self.temporal_axis]
        out_seqlen = inp_timeline[-1] + 1  # assuming timeline must include the last time step
        assert len(inp_timeline) == orig_seqlen
        out_timeline = np.arange(out_seqlen)
        if orig_seqlen == out_seqlen:
            print('WARNING - Interpolator: interpolation was not operated.')
            return inp

        # interpolote
        interped = {}
        for k, v in inp.items():
            interp_fn = interp1d(inp_timeline, v.cpu().numpy(), axis=self.temporal_axis, kind=self.interp_type)
            interped[k] = interp_fn(out_timeline)
            # print(interped.shape)
            for i in range(len(v.shape)):
                if i == self.temporal_axis:
                    assert interped[k].shape[i] == out_seqlen
                else:
                    assert interped[k].shape[i] == v.shape[i]
            interped[k] = torch.tensor(interped[k], device=v.device, dtype=torch.float32)
        return interped


class DiffInterpolator(nn.Module):
    # differentiable interpolation, based on torchinterp1d
    # for differentiability, we assume x_sample is sorted
    def __init__(
            self,
            interp_type='linear',
            sample_type='non_adaptive',
            temporal_axis=1,
    ):
        super(DiffInterpolator, self).__init__()
        if temporal_axis != 1:
            raise ValueError('currently supporting only temporal_axis=1 (got {})'.format(temporal_axis))
        if interp_type not in ['linear']:
            raise ValueError('Unsupported interp_type [{}]'.format(interp_type))
        if sample_type not in ['non-adaptive']:
            raise ValueError('Unsupported interp_type [{}]'.format(sample_type))

        self.interp_type = interp_type
        self.sample_type = sample_type
        self.temporal_axis = temporal_axis
        self.interpolator = Interp1d()
        print('Diff Interpolator - running [{}, {}]'.format(self.interp_type, self.sample_type))

    def forward(self, inp, inp_timeline):

        # inp_timeline -> x
        # inp -> y
        # out_timeline -> x_new
        # interped -> y_new

        orig_seqlen = list(inp.values())[0].shape[self.temporal_axis]
        out_seqlen = inp_timeline[-1] + 1  # assuming timeline must include the last time step
        assert len(inp_timeline) == orig_seqlen
        out_timeline = torch.arange(out_seqlen).to(list(inp.values())[0].device)
        if orig_seqlen == out_seqlen:
            print('WARNING - Interpolator: interpolation was not operated.')
            return inp

        # interpolate
        # print(inp.keys())
        interped = {}
        for k in inp.keys():
            # y_pred = self.interp(x_hat, y_hat, x_new)
            # print((inp_timeline.shape, type(inp_timeline)))
            # print((inp[k].shape, type(inp[k])))
            # print((out_timeline.shape, type(out_timeline)))
            # print(k)
            interped[k] = self.interp(inp_timeline, inp[k], out_timeline)
            # print(interped[k].shape)
            expected_shape = list(inp[k].shape)
            expected_shape[self.temporal_axis] = out_seqlen
            assert interped[k].shape == torch.Size(expected_shape)
        return interped

    def interp(self, x, y, xnew):
        # A wrapper for interp1d call
        # print(y.shape)
        orig_shape = y.shape   # [bs, in_seqlen, f]
        in_seqlen = orig_shape[self.temporal_axis]
        out_seqlen = xnew.shape[0]

        out_shape = list(orig_shape)
        out_shape[self.temporal_axis] = out_seqlen  # [bs, out_seqlen, [f]]
        out_shape = torch.Size(out_shape)

        out_shape_before_swap = list(out_shape) # [bs, [f], out_seqlen]
        out_shape_before_swap[self.temporal_axis] = out_shape[-1]
        out_shape_before_swap[-1] = out_shape[self.temporal_axis]
        out_shape_before_swap = torch.Size(out_shape_before_swap)

        # print('out_shape_before_swap: {}'.format(out_shape_before_swap))
        # print('out_shape: {}'.format(out_shape))
        # print('out_seqlen: {}'.format(out_seqlen))
        #
        # print('GUYGUY1:y {}'.format(y.shape))
        _y = torch.transpose(y, self.temporal_axis, -1)
        _y = _y.reshape(-1, in_seqlen)
        _x = torch.tile(x.view(1, -1), (_y.shape[0], 1))
        _xnew = torch.tile(xnew.view(1, -1), (_y.shape[0], 1))
        # print('GUYGUY2:_y {}'.format(_y.shape))
        # print('GUYGUY2:_x {}'.format(_x.shape))
        # print('GUYGUY2:_xnew {}'.format(_xnew.shape))
        # ynew = self.interpolator(x, _y.view(-1, orig_seqlen), xnew)
        ynew = self.interpolator(_x, _y, xnew).view(out_shape_before_swap)
        # print('GUYGUY3: {}'.format(ynew.shape))
        return torch.transpose(ynew, -1, self.temporal_axis)


class GeometricProcess(nn.Module):
    def __init__(
            self
    ):
        super(GeometricProcess, self).__init__()

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

    def forward(self, pred, J_regressor=None):

        bs, seqlen, _ = pred['cam'].shape
        flat_bs = bs * seqlen

        # flatten
        flat = {}
        flat['pose'] = pred['pose'].reshape(flat_bs, -1, 6)
        flat['shape'] = pred['shape'].reshape(flat_bs, 10)
        flat['cam'] = pred['cam'].reshape(flat_bs, 3)

        pred_rotmat = rot6d_to_rotmat(flat['pose']).view(flat_bs, 24, 3, 3)
        # print(pred_rotmat.device)
        # for k, v in pred.items():
        #     print('{}: {}, {}'.format(k, v.shape, v.device))
        pred_output = self.smpl(
            betas=flat['shape'],
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, flat['cam'])

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        smpl_output = {
            'theta'  : torch.cat([flat['cam'], pose, flat['shape']], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }

        smpl_output['theta'] = smpl_output['theta'].reshape(bs, seqlen, -1)
        smpl_output['verts'] = smpl_output['verts'].reshape(bs, seqlen, -1, 3)
        smpl_output['kp_2d'] = smpl_output['kp_2d'].reshape(bs, seqlen, -1, 2)
        smpl_output['kp_3d'] = smpl_output['kp_3d'].reshape(bs, seqlen, -1, 3)
        smpl_output['rotmat'] = smpl_output['rotmat'].reshape(bs, seqlen, -1, 3, 3)

        return smpl_output

