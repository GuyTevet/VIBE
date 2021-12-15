import torch
# from torchinterp1d import Interp1d
from lib.utils.interp1d import Interp1d
import os
from tqdm import tqdm
import pickle
import argparse

torch.autograd.set_detect_anomaly(True)
device = 'cpu'

class ValOffsetModel(torch.nn.Module):
    def __init__(self, model_type, val_params, offset_params):
        super(ValOffsetModel, self).__init__()
        assert val_params['min'] < val_params['max']
        assert offset_params['min'] < offset_params['max']
        self.val_params = val_params
        self.offset_params = offset_params
        self.output_size = val_params['size'] + offset_params['size']
        self.model_type = model_type

        if self.model_type == 'mlp':
            width=25
            self.lin1 = torch.nn.Linear(width, width)
            self.lin2 = torch.nn.Linear(width, self.output_size)
        elif self.model_type == 'gru':
            n_layers = 2
            bs = 1
            h_size = 10
            width = val_params['size']
            self.gru = torch.nn.GRU(input_size=1, hidden_size=h_size, num_layers=n_layers)
            self.lin = torch.nn.Linear(h_size, 2)
            self.h0 = torch.randn(n_layers, bs, h_size).to(device)
        else:
            raise ValueError('unsupported model type [{}]'.format(self.model_type))

        self.relu = torch.nn.ReLU(inplace=True)
        self.sig = lambda x, _min, _max: torch.sigmoid(x) * (_max - _min) + _min
        self.input_noise = torch.rand(width).to(device)
        if self.model_type == 'gru':
            self.input_noise = self.input_noise.reshape(self.input_noise.shape[0], 1, 1)
            

    def forward(self):
        if self.model_type == 'mlp':
            out = self.lin1(self.input_noise)
            out = self.relu(out)
            out = self.lin2(out)
            val = out[:self.val_params['size']]
            offset = out[self.val_params['size']:]
        elif self.model_type == 'gru':
            out, _ = self.gru(self.input_noise, self.h0)  # [n_samples, 1, h_size]
            # out = torch.zeros([self.input_noise.shape[0], 1, 10]).cuda()
            # out = torch.squeeze(out)
            out = self.relu(out)
            out = self.lin(out.view(-1, out.size(-1)))
            val = out[: ,0]
            offset = out[1:-1, 1]

        return {
            'val': self.sig(val, self.val_params['min'], self.val_params['max']),
            'offset': self.sig(offset, self.offset_params['min'], self.offset_params['max']),
        }


class Interpolator(torch.nn.Module):
    def __init__(self, model_type, method, n_samples, amp, min_offset, max_offset):
        super(Interpolator, self).__init__()
        self.method = method
        self.n_samples = n_samples
        self.model = ValOffsetModel(model_type,
                                    val_params={'min': -amp,  'max': amp, 'size': n_samples},
                                    offset_params={'min': min_offset,  'max': max_offset, 'size': n_samples-2})
        self.interp = Interp1d()

    def forward(self, x_new, x_sampled):
        res = self.model()

        if self.method == 'no_sampling':
            return {'y_pred': res['val']}
        else:
            y_hat = res['val']
            offsets = res['offset']
            offsets = torch.cat((torch.tensor([0.]).to(device), offsets, torch.tensor([0.]).to(device)))
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
                # assert x_hat[-1] > x_hat[-2]
            else:
                raise ValueError('Invalid method [{}]'.format(self.method))
            y_pred = self.interp(x_hat, y_hat, x_new)
            return {'y_pred': y_pred, 'x_hat': x_hat, 'y_hat': y_hat}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mlp', choices=['mlp', 'gru'], help='')
    parser.add_argument("--signal", type=str, default='signal_noise', choices=['signal_noise', 'chirp'], help='')
    parser.add_argument("--sample_rate", type=int, default=4, help='')
    params = parser.parse_args()

    # define constants
    seqlen = 100
    num_samples = seqlen // params.sample_rate  # 25
    bs = 32
    num_batches = 20
    max_value = 2
    n_steps = 10000 if params.model_type == 'gru' else 1000

    out_path = './visuals/interp_1d_{}_exp_{}_samplerate{}.pkl'.format(params.signal, params.model_type, params.sample_rate)
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    # define GT
    signals = {
        'chirp': lambda x: torch.sin(2*3.14*((1./seqlen)* x + (1./(seqlen*40)*torch.pow(x, 2)))) * max_value,
        'signal_noise': lambda x: (torch.sin(2*3.14*((1./seqlen)* x)) + (0.2*x/seqlen) * torch.sin(2*3.14*((40./seqlen)* x))) * max_value,
    }

    x_new = torch.arange(seqlen, dtype=torch.float).to(device)
    x_sampled = x_new[0::params.sample_rate].to(device)
    if x_sampled[-1] != seqlen-1:
        x_sampled = torch.cat((x_sampled.to(device), torch.tensor([seqlen-1]).to(device)))
    y_new = signals[params.signal](x_new).to(device)

    # define model
    models ={
        'no_sampling': Interpolator(params.model_type, 'no_sampling', seqlen, max_value, min_offset=-1, max_offset=1).to(device), # offsets ignored
        'non_adaptive': Interpolator(params.model_type, 'non_adaptive', len(x_sampled), max_value, min_offset=-1, max_offset=1).to(device), # offsets ignored
        'adaptive': Interpolator(params.model_type, 'adaptive', len(x_sampled), max_value, min_offset=-(params.sample_rate / 2), max_offset=(params.sample_rate / 2)).to(device),
        'super_adaptive': Interpolator(params.model_type, 'super_adaptive', len(x_sampled), max_value, min_offset=10e-3, max_offset=2*params.sample_rate).to(device),
    }
    opts = {name: torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for name, model in models.items()}
    loss = torch.nn.MSELoss().to(device)
    n_params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}

    print('#Parameters:')
    for name, p in n_params.items():
        print('{}: {}'.format(name, p))

    tensor2numpy = lambda t: t.detach().cpu().numpy()
    dict2numpy = lambda d: {k: tensor2numpy(v) for k, v in d.items()}
    ddict2numpy = lambda dd: {k: dict2numpy(d) for k, d in dd.items()}

    buffer = {
        'sample_rate': params.sample_rate,
        'model_type': params.model_type,
        'model_names': list(models.keys()),
        'n_params': n_params,
        'x_new': tensor2numpy(x_new),
        'y_new': tensor2numpy(y_new),
        'x_sampled': tensor2numpy(x_sampled),
        'train': [],

    }

    for step_i in tqdm(range(n_steps)):
        preds = {name: model(x_new, x_sampled) for name, model in models.items()}
        errs = {name: loss(preds[name]['y_pred'], y_new) for name in models.keys()}

        for name in models.keys():
            opts[name].zero_grad()
            errs[name].backward()
            opts[name].step()

        # buffer training state
        if step_i % (n_steps // 100) == 0:
            # noinspection PyTypeChecker
            buffer['train'].append({
                'step': step_i,
                'preds': ddict2numpy(preds),
                'errs': dict2numpy(errs),
            })

    # Save pkl
    with open(out_path, 'wb') as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
