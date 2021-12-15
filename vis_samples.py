import matplotlib.pyplot as plt
import json
import os
import pickle
import numpy as np

# Added manually according to https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
SMPL_JOINT_NAMES = [
    'Pelvis',  # 0
    'L_Hip',  # 1
    'R_Hip',  # 2
    'Spine1',  # 3
    'L_Knee',  # 4
    'R_Knee',  # 5
    'Spine2',  # 6
    'L_Ankle',  # 7
    'R_Ankle',  # 8
    'Spine3',  # 9
    'L_Foot',  # 10
    'R_Foot',  # 11
    'Neck',  # 12
    'L_Collar',  # 13
    'R_Collar',  # 14
    'Head',  # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',  # 18
    'R_Elbow',  # 19
    'L_Wrist',  # 20
    'R_Wrist',  # 21
    'L_Hand',  # 22
    'R_Hand',  # 23
]

if __name__ == '__main__':
    exp_name = 'interp_linear'
    samples_dir = 'samples'
    results_dir = os.path.join(samples_dir, 'vis', exp_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    gt_pkl = os.path.join(samples_dir, 'gt.pkl')
    baseline_pkl = os.path.join(samples_dir, 'seqlen_64_samples.pkl')
    models_pkl = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.startswith(exp_name) and f.endswith('.pkl')]
    params = [int(os.path.basename(f).replace('_samples.pkl', '').split('_')[-1]) for f in models_pkl]

    # read files
    data = {}
    with open(gt_pkl, "rb") as f:
        gt = pickle.load(f, encoding="latin1")
    with open(baseline_pkl, "rb") as f:
        data['baseline'] = pickle.load(f, encoding="latin1")
    for i, pkl in enumerate(models_pkl):
        with open(pkl, "rb") as f:
            data['model_{}'.format(params[i])] = pickle.load(f, encoding="latin1")
    n_samples = 5
    seqlen = gt['thetas'][0].shape[0] # 64
    bs = len(gt['thetas'])

    modes = ['kp_3d', 'thetas']
    for bs_i in range(bs):
        gt['thetas'][bs_i] = gt['thetas'][bs_i].reshape((seqlen, -1, 3))
        for m in data.keys():
            data[m]['thetas'][bs_i] = data[m]['thetas'][bs_i].reshape((seqlen, -1, 3))

    for mode in modes:
        joint_dim = gt[mode][0].shape[1]  # 14
        for joint_i in range(joint_dim):
            fig, axs = plt.subplots(3, n_samples, figsize=(6 * n_samples, 4*3))
            fig.tight_layout()
            if mode == 'thetas':
                fig.suptitle('{} | {} |Joint [{}, {}]'.format(exp_name, mode, SMPL_JOINT_NAMES[joint_i], joint_i))
            else:
                fig.suptitle('{} | {} |Joint [{}]'.format(exp_name, mode, joint_i))
            save_path = os.path.join(results_dir, '{}_{}_joint_{}.png'.format(exp_name, mode, joint_i))
            for sample_i in range(n_samples):
                for i, xyz in enumerate(['X', 'Y', 'Z']):
                    # axs[sample_i].set_title(model_name)
                    axs[i, sample_i].plot(np.arange(seqlen), gt[mode][sample_i][:, joint_i, i], label='GT signal')
                    for model_i, model_name in enumerate(data.keys()):
                        axs[i, sample_i].plot(np.arange(seqlen), data[model_name][mode][sample_i][:, joint_i, i], label=model_name) # , linestyle='-', marker='o')
                    if sample_i == n_samples-1:
                        axs[i, sample_i].legend()
            fig.savefig(save_path)


