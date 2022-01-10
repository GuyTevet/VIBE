import os
import cv2
import glob
import h5py
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio
import matplotlib.pyplot as plt
import clip
import torch
from PIL import Image

import sys
sys.path.append(".")

from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d
from lib.data_utils.feature_extractor import extract_features

# Guy: This scripts generates db file for the Motion-Clip experiment
# The examples are (Clip_i, Joint2D) pairs

def process_image_kp2d(image_path, kp2d, cube_dim):
    """
    image_path: image full path
    kp2d: 2d keypoints [17, 2]
    cube_dim: clip image input hight and width
    """
    im = plt.imread(image_path)
    ul = np.array([max(0, kp2d[:, 0].min()), max(0, kp2d[:, 1].min())]).squeeze().astype(
        int)  # upper left
    lr = np.array([min(kp2d[:, 0].max(), im.shape[1] - 1),
                   min(kp2d[:, 1].max(), im.shape[0] - 1)]).squeeze().astype(int)  # lower right
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    cubic_dim = max(h, w)
    x_bias = (cubic_dim - w) // 2
    y_bias = (cubic_dim - h) // 2
    im_crop = np.zeros((cubic_dim, cubic_dim, 3), dtype=im.dtype)
    im_crop[y_bias:y_bias + h, x_bias: x_bias + w, :] = im[ul[1]: lr[1], ul[0]: lr[0], :]
    im_crop_resize = cv2.resize(im_crop, dsize=(cube_dim, cube_dim), interpolation=cv2.INTER_CUBIC)
    joints_2d_crop = kp2d - ul + [x_bias, y_bias]
    joints_2d_crop_normal = joints_2d_crop / cubic_dim
    return im_crop_resize, joints_2d_crop_normal



def read_openpose(json_file, gt_part, dataset):
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                dist_conf[i] = np.mean(np.sqrt(np.sum(op_conf12*(op_keyp12 - gt_part[:12, :2])**2, axis=1)))
        # closest match
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        if dataset == 'mpii':
            thresh = 30
        elif dataset == 'coco':
            thresh = 10
        else:
            thresh = 0
        # dataset-specific thresholding based on pixel size of person
        if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
            keyp25 = np.zeros([25,3])
        else:
            keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def read_data_train(dataset_path, debug=False):
    h, w = 2048, 2048
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    # training data
    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    # product = product(user_list, seq_list, vid_list)
    # user_i, seq_i, vid_i = product[process_id]

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            for j, vid_i in enumerate(vid_list):
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'video_' + str(vid_i))
                # per frame
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                vid_used_frames = []
                vid_used_joints = []
                vid_used_bbox = []
                vid_segments = []
                vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i) + "_vid" + str(vid_i) + "_seg0"
                for i, img_i in tqdm_enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    joints_2d_raw = np.reshape(annot2[vid_i][0][i], (1, 28, 2))
                    joints_2d_raw= np.append(joints_2d_raw, np.ones((1,28,1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "mpii3d",  "spin").reshape((-1,3))

                    # visualize = True
                    # if visualize == True and i == 500:
                    #     import matplotlib.pyplot as plt
                    #
                    #     frame = cv2.cvtColor(cv2.imread(img_i), cv2.COLOR_BGR2RGB)
                    #
                    #     for k in range(49):
                    #         kp = joints_2d[k]
                    #
                    #         frame = cv2.circle(
                    #             frame.copy(),
                    #             (int(kp[0]), int(kp[1])),
                    #             thickness=3,
                    #             color=(255, 0, 0),
                    #             radius=5,
                    #         )
                    #
                    #         cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    #                     (0, 255, 0),
                    #                     thickness=3)
                    #
                    #     plt.imshow(frame)
                    #     plt.show()

                    joints_3d_raw = np.reshape(annot3[vid_i][0][i], (1, 28, 3)) / 1000
                    joints_3d = convert_kps(joints_3d_raw, "mpii3d", "spin").reshape((-1,3))

                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    # check that all joints are visible
                    x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                    y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < joints_2d.shape[0]:
                        vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                                          str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                        continue

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(img_name.split(".")[0])
                    dataset['img_name'].append(img_i)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_segments.append(vid_uniq_id)
                    vid_used_frames.append(img_i)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

                vid_segments= np.array(vid_segments)
                ids = np.zeros((len(set(vid_segments))+1))
                ids[-1] = len(vid_used_frames) + 1
                if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                    ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

                for i in tqdm(range(len(set(vid_segments)))):
                    features = extract_features(model, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                                vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                                kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                                dataset='spin', debug=False)
                    dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


def read_test_data(dataset_path, clip_name, debug=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_name, device)

    dataset = {
        # 'vid_name': [],
        # 'frame_id': [],
        # 'joints3D': [],
        'joints2D': [],
        # 'bbox': [],
        'img_name': [],
        'features': [],
        "valid_i": []
    }

    debug_path = 'data/debug_mpii'
    if debug and not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # model = spin.get_pretrained_hmr()

    user_list = range(1, 7)

    for user_i in user_list:
        print('Subject', user_i)
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])

        # vid_used_frames = []
        # vid_used_joints = []
        # vid_used_bbox = []
        # vid_segments = []
        # vid_uniq_id = "subj" + str(user_i) + "_seg0"
        processed_images = []
        for frame_i, valid_i in tqdm(enumerate(valid[:200])):

            img_i = os.path.join('mpi_inf_3dhp_test_set',
                                    'TS' + str(user_i),
                                    'imageSequence',
                                    'img_' + str(frame_i + 1).zfill(6) + '.jpg')

            joints_2d_raw = annot2[frame_i, 0, :, :]
            if any(joints_2d_raw.max(0)<0):
                print('Skip frame [{}]'.format(frame_i))
                continue
            # joints_2d_raw = np.append(joints_2d_raw, np.ones((1, 17, 1)), axis=2)

            # joints_3d_raw = np.reshape(annot3[frame_i, 0, :, :], (1, 17, 3)) / 1000
            # joints_3d = convert_kps(joints_3d_raw, "mpii3d_test", "spin").reshape((-1, 3))
            # joints_3d = joints_3d - joints_3d[39] # substract pelvis zero is the root for test
            #
            # bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

            clip_image_dim = preprocess.transforms[1].size[0] # 224
            img_file = os.path.join(dataset_path, img_i)
            im_crop_resize, joints_2d_crop_normal = process_image_kp2d(img_file, joints_2d_raw, clip_image_dim)
            processed_images.append(torch.tensor(preprocess(Image.fromarray(im_crop_resize)), device=device).unsqueeze(0))

            ###### DEBUG
            if debug and frame_i % 100 == 0:
                plt.imshow(im_crop_resize)
                plt.scatter(x=joints_2d_crop_normal[:, 0] * clip_image_dim, y=joints_2d_crop_normal[:, 1] * clip_image_dim, c='r', s=5)
                plt.axis('off')
                save_path = os.path.join(debug_path, 'user{}_frame{}.jpg'.format(user_i, frame_i))
                plt.savefig(save_path)
                plt.close()

            # # check that all joints are visible
            # img_file = os.path.join(dataset_path, img_i)
            # I = cv2.imread(img_file)
            # h, w, _ = I.shape
            # x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
            # y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
            # ok_pts = np.logical_and(x_in, y_in)
            #
            # if np.sum(ok_pts) < joints_2d.shape[0]:
            #     vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1]) + "_seg" + \
            #                   str(int(dataset['vid_name'][-1].split("_")[-1][3:]) + 1)
            #     continue



            # dataset['vid_name'].append(vid_uniq_id)
            # dataset['frame_id'].append(img_file.split("/")[-1].split(".")[0])
            dataset['img_name'].append(img_file)
            # dataset['joints2D'].append(joints_2d)
            dataset['joints2D'].append(joints_2d_crop_normal)
            # dataset['joints3D'].append(joints_3d)
            # dataset['bbox'].append(bbox)
            dataset['valid_i'].append(valid_i)

            # vid_segments.append(vid_uniq_id)
            # vid_used_frames.append(img_file)
            # vid_used_joints.append(joints_2d)
            # vid_used_bbox.append(bbox)

        # vid_segments = np.array(vid_segments)
        # ids = np.zeros((len(set(vid_segments)) + 1))
        # ids[-1] = len(vid_used_frames) + 1
        # if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
        #     ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

        # for i in tqdm(range(len(set(vid_segments)))):
        #     features = extract_features(model, np.array(vid_used_frames)[int(ids[i]):int(ids[i + 1])],
        #                                 vid_used_bbox[int(ids[i]):int(ids[i + 1])],
        #                                 kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i + 1])],
        #                                 dataset='spin', debug=False)
        #     dataset['features'].append(features)


        # Apply CLIP
        batch_size = 64
        # Prepare the inputs
        image_features = []
        for batch_i in tqdm(range(len(processed_images) // batch_size + (len(processed_images) % batch_size > 0))):
            images = torch.cat(processed_images[batch_i*batch_size : (batch_i+1)*batch_size], axis=0)
            with torch.no_grad():
                image_features.append(model.encode_image(images)) # [bs, 512]

        image_features = [t.squeeze().cpu().numpy() for t in torch.cat(image_features, axis=0).split(1)]
        assert len(image_features) == len(processed_images)
        dataset['features'] += image_features

    # for i in tqdm(range(len(set(vid_segments)))):
    #     features = extract_features(model, np.array(vid_used_frames)[int(ids[i]):int(ids[i + 1])],
    #                                 vid_used_bbox[int(ids[i]):int(ids[i + 1])],
    #                                 kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i + 1])],
    #                                 dataset='spin', debug=False)
    #     dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/mpii_3d')
    parser.add_argument('--clip', type=str, help='clip model type', default='ViT-B/32',
                        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'])
    args = parser.parse_args()

    clip_db = 'data/motion_clip'
    if not os.path.exists(clip_db):
        os.makedirs(clip_db)

    dataset = read_test_data(args.dir, args.clip, debug=False)
    joblib.dump(dataset, osp.join(clip_db, 'mpii3d_clip_{}_val_db.pt'.format(args.clip).replace('/', '')))

    # dataset = read_data_train(args.dir, args.clip, debug=False)
    # joblib.dump(dataset, osp.join(clip_db, 'mpii3d_clip_{}_train_db.pt'.format(args.clip)))



