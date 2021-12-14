import matplotlib.pyplot as plt
import json
import os
import pickle
import numpy as np

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
    joint_dim = gt['thetas'][0].shape[-1] # 72
    seqlen = gt['thetas'][0].shape[0] # 64

    for joint_i in range(joint_dim):
        fig, axs = plt.subplots(n_samples, len(data.keys()), figsize=(6 * len(data.keys()), 6 * n_samples))
        fig.suptitle('{} | Joint index [{}]'.format(exp_name, joint_i))
        save_path = os.path.join(results_dir, '{}_joint_{}.png'.format(exp_name, joint_i))
        for sample_i in range(n_samples):
            for model_i, model_name in enumerate(data.keys()):
                axs[sample_i, model_i].set_title(model_name)
                axs[sample_i, model_i].plot(np.arange(seqlen), gt['thetas'][sample_i][:, joint_i], label='GT signal')
                axs[sample_i, model_i].plot(np.arange(seqlen), data[model_name]['thetas'][sample_i][:, joint_i],
                                            linestyle='-', marker='o')
        fig.savefig(save_path)

