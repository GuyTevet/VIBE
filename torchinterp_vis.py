import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torchinterp1d import Interp1d
import os
import shutil
import cv2
from tqdm import tqdm

class ToyModel(torch.nn.Module):
    def __init__(self, _min, _max, width):
        super(ToyModel, self).__init__()
        assert _min < _max
        self.min = _min
        self.max = _max
        self.lin1 = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU(inplace=True)
        self.lin2 = torch.nn.Linear(width, width)
        self.sig = lambda x: torch.sigmoid(x)*(self.max-self.min) + self.min  # (-1, 1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.sig(out)
        return out

class Interpolator(torch.nn.Module):
    def __init__(self, method, n_samples, amp, min_offset, max_offset):
        super(Interpolator, self).__init__()
        self.method = method
        self.n_samples = n_samples
        self.offset_model = ToyModel(min_offset, max_offset, width=n_samples-2)
        self.value_model = ToyModel(-amp, amp, width=n_samples)
        self.input_noise = torch.rand(n_samples)
        self.interp = Interp1d()

    def forward(self, x_new, x_sampled):
        if self.method == 'no_sampling':
            return {'y_pred': self.value_model(self.input_noise)}
        else:
            y_hat = self.value_model(self.input_noise)
            offsets = self.offset_model(self.input_noise[:-2])
            offsets = torch.cat((torch.tensor([0.]), offsets, torch.tensor([0.])))
            if self.method == 'non_adaptive':
                x_hat = x_sampled
            elif self.method == 'adaptive':
                x_hat = x_sampled + offsets
            elif self.method == 'super_adaptive':
                offsets[-2] /= 2.  # FIXME - ugly hack for super-adaptive case (avoiding the assertion)
                x_hat = torch.zeros_like(y_hat)
                for i in range(1, len(x_sampled) - 1):
                    x_hat[i] = x_hat[i - 1] + offsets[i]
                x_hat[-1] = seqlen - 1
                assert x_hat[-1] > x_hat[-2]
            else:
                raise ValueError('Invalid method [{}]'.format(self.method))
            y_pred = self.interp(x_hat, y_hat, x_new)
            return {'y_pred': y_pred, 'x_hat': x_hat, 'y_hat': y_hat}


if __name__ == "__main__":

    # define constants
    seqlen = 100
    sample_rate = 4
    num_samples = seqlen // sample_rate  # 25
    bs = 32
    num_batches = 20
    max_value = 2

    plot_dir = '/tmp/interp_1d_exp_plt'
    out_path = '../visuals/interp/interp_1d_exp_samplerate{}.mp4'.format(sample_rate)
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)


    # define GT
    chirp_fn = lambda x: torch.sin(2*3.14*((1./seqlen)* x + (1./(seqlen*40)*torch.pow(x, 2)))) * max_value
    x_new = torch.arange(seqlen, dtype=torch.float)
    x_sampled = x_new[0::sample_rate]
    if x_sampled[-1] != seqlen-1:
        x_sampled = torch.cat((x_sampled, torch.tensor([seqlen-1])))
    y_new = chirp_fn(x_new)

    # define model
    models ={
        'no_sampling': Interpolator('no_sampling', seqlen, max_value, min_offset=-1, max_offset=1), # offsets ignored
        'non_adaptive': Interpolator('non_adaptive', len(x_sampled), max_value, min_offset=-1, max_offset=1), # offsets ignored
        'adaptive': Interpolator('adaptive', len(x_sampled), max_value, min_offset=-(sample_rate / 2), max_offset=(sample_rate / 2)),
        'super_adaptive': Interpolator('super_adaptive', len(x_sampled), max_value, min_offset=10e-3, max_offset=2*sample_rate),
    }
    opts = {name: torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for name, model in models.items()}
    loss = torch.nn.MSELoss()

    for step_i in tqdm(range(1000)):
        preds = {name: model(x_new, x_sampled) for name, model in models.items()}
        errs = {name: loss(preds[name]['y_pred'], y_new) for name in models.keys()}

        for name in models.keys():
            opts[name].zero_grad()
            errs[name].backward()
            opts[name].step()

        # plot one frame
        if step_i % 10 == 0:
            fig, axs = plt.subplots(1, len(models.keys()), figsize=(24, 6))
            fig.suptitle('Sample rate [{}] | Step [{}]'.format(sample_rate, step_i))

            for ax_i, name in enumerate(models.keys()):
                axs[ax_i].set_title(name + ' - MSE[{:.4f}]'.format(errs[name].detach()))
                axs[ax_i].plot(x_new, y_new, label='GT signal')
                if name == 'no_sampling':
                    axs[ax_i].plot(x_new, preds[name]['y_pred'].detach(), linestyle='-', marker='o', color='green')
                else:
                    axs[ax_i].plot(x_sampled, y_new[x_sampled.to(torch.long)], linestyle='None', marker='v')
                    axs[ax_i].plot(preds[name]['x_hat'].detach(), preds[name]['y_hat'].detach(), linestyle='-', marker='o')
            plt.savefig(os.path.join(plot_dir, 'step_{:05d}.png'.format(step_i)))
            plt.close()


    # save clip
    images = [os.path.join(plot_dir, f) for f in os.listdir(plot_dir) if f.endswith('.png')]
    images.sort()
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(out_path, fourcc, 5., (width, height))
    for image in tqdm(images):
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
