import pickle
import torch
import os
import shutil
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    input_pkl = '../visuals/interp/interp_1d_exp_mlp_samplerate4.pkl'
    out_mp4 = input_pkl.replace('.pkl', '.mp4')
    plot_dir = '/tmp/interp_1d_exp_plt'
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    with open(input_pkl, 'rb') as handle:
        buffer = pickle.load(handle)

    for snapshot in tqdm(buffer['train']):
        # plot one frame
        if snapshot['step'] % 10 == 0:
            fig, axs = plt.subplots(1, len(buffer['model_names']), figsize=(24, 6))
            fig.suptitle('{} | Sample rate [{}] | Step [{}]'.format(buffer['model_type'], buffer['sample_rate'], snapshot['step']))
    
            for ax_i, name in enumerate(buffer['model_names']):
                axs[ax_i].set_title(name + '- #Params[{}] - MSE[{:.4f}]'.format(buffer['n_params'][name], snapshot['errs'][name]))
                axs[ax_i].plot(buffer['x_new'], buffer['y_new'], label='GT signal')
                if name == 'no_sampling':
                    axs[ax_i].plot(buffer['x_new'], snapshot['preds'][name]['y_pred'], linestyle='-', marker='o', color='green')
                else:
                    axs[ax_i].plot(buffer['x_sampled'], buffer['y_new'][buffer['x_sampled'].astype(int)], linestyle='None', marker='v')
                    axs[ax_i].plot(snapshot['preds'][name]['x_hat'], snapshot['preds'][name]['y_hat'], linestyle='-', marker='o')
            plt.savefig(os.path.join(plot_dir, 'step_{:05d}.png'.format(snapshot['step'])))
            plt.close()

    # save clip
    images = [os.path.join(plot_dir, f) for f in os.listdir(plot_dir) if f.endswith('.png')]
    images.sort()
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(out_mp4, fourcc, 5., (width, height))
    for image in tqdm(images):
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()