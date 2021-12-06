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
        timeline = np.arange(seqlen)
        sample_timeline = timeline[0::self.dilation_rate]
        if sample_timeline[-1] != timeline[-1]:
            sample_timeline = np.append(sample_timeline, timeline[-1])
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

