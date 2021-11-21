import torch
import os
import torch
import os.path as osp
import torch.nn as nn
import numpy as np

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
        return out