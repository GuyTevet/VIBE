# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import math
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torchvision.models.resnet as resnet

from lib.core.config import VIBE_DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS


class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HMR(nn.Module):
    """
    SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        ).to('cpu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def feature_extractor(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        return xf

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_features=False):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
        }]

        if return_features:
            return xf, output
        else:
            return output


class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, interp_type='linear', interp_ratio=None):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.interp_type = interp_type
        self.interp_ratio = interp_ratio
        if interp_ratio is not None:
            print('Regressor will execute interp [{}, {}]'.format(self.interp_type, self.interp_ratio))
        else:
            print('Regressor - No interp!')

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    # TODO - insert interpolator fn here
    def interp_batch(self, data, interp_type, interp_ratio):
        import numpy as np
        from scipy.interpolate import interp1d
        temporal_axis = 1
        # print(batch.keys())
        # for k in batch.keys():
        #     print('{}: {}'.format(k, data.shape))
        # print(interp_ratio)
        # print(interp_ratio is not None)
        assert type(interp_ratio) == int

        # sample
        timeline = np.arange(data.shape[temporal_axis])
        sample_timeline = timeline[0::interp_ratio]
        if sample_timeline[-1] != timeline[-1]:
            sample_timeline = np.append(sample_timeline, timeline[-1])
        sample = data[:, sample_timeline, ...].cpu().numpy() # temporal_axis=1

        # print(timeline)
        # print(sample_timeline)
        # print(sample.shape)
        # print(len(sample.shape)-1)

        # interpoloate
        interp_fn = interp1d(sample_timeline, sample, axis=temporal_axis, kind=interp_type)
        interped = interp_fn(timeline)

        # scale_factor = tuple([1.] * (len(data.shape) - 1 - 2) + [float(interp_ratio)])
        # interped = torch.nn.functional.interpolate(sample, scale_factor=scale_factor, mode=interp_type,
        #                                            align_corners=None, recompute_scale_factor=None)
        # print(interped.shape)

        assert interped.shape == data.shape
        return torch.tensor(interped, device=data.device, dtype=torch.float32)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None, orig_bs=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        # Interpolation module
        if self.interp_ratio is not None:
            assert type(self.interp_ratio) == int
            import pytorch3d.transforms as T

            # _pose = T.matrix_to_euler_angles(pred_rotmat, 'XYZ')
            _pose = T.matrix_to_rotation_6d(pred_rotmat)
            assert orig_bs is not None
            # orig_bs = 32  # FIXME - avoid hardcoding - for eval
            # orig_bs = 1  # FIXME - avoid hardcoding - for demo
            bs, n_joints, n_dims = _pose.shape
            seqlen = bs//orig_bs
            print('DEBUG: Considering seqlen [{}]'.format(seqlen)) # just for validation

            # to be interped
            _pose = _pose.reshape(orig_bs, seqlen, n_joints, n_dims)
            _shape = pred_shape.reshape(orig_bs, seqlen, 10)
            _cam = pred_cam.reshape(orig_bs, seqlen, 3)

            # interp
            # interp_type = 'cubic'  # FIXME - hardcoded interp_type, interp_ratio
            # interp_ratio = 32
            _pose = self.interp_batch(_pose, self.interp_type, self.interp_ratio)
            _shape = self.interp_batch(_shape, self.interp_type, self.interp_ratio)
            _cam = self.interp_batch(_cam, self.interp_type, self.interp_ratio)

            # override original data
            _pose = _pose.reshape(batch_size, n_joints, n_dims)
            # _pose = T.euler_angles_to_matrix(_pose, 'XYZ')
            _pose = T.rotation_6d_to_matrix(_pose)
            assert _pose.shape == pred_rotmat.shape
            pred_rotmat = _pose
            pred_shape = _shape.reshape(batch_size, 10)
            pred_cam = _cam.reshape(batch_size, 3)

        pred_output = self.smpl(
            betas=pred_shape,
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

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output


def hmr(smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True, **kwargs):
    """
    Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(), strict=False)
    return model


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


def get_pretrained_hmr():
    device = 'cuda'
    model = hmr().to(device)
    checkpoint = torch.load(osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model
