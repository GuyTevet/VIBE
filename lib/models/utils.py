import torch
import os
import torch
import os.path as osp
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d


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
        return out, timeline

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
        interp_fn = interp1d(inp_timeline, inp.cpu().numpy(), axis=self.temporal_axis, kind=self.interp_type)
        interped = interp_fn(out_timeline)
        # print(interped.shape)

        for i in range(len(inp.shape)):
            if i == self.temporal_axis:
                assert interped.shape[i] == out_seqlen
            else:
                assert interped.shape[i] == inp.shape[i]
        return torch.tensor(interped, device=inp.device, dtype=torch.float32)