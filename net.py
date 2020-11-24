import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

_BASE_CHANNELS = 64

def init_conv(conv):
    torch.nn.init.kaiming_normal_(conv.weight, a=4.4, nonlinearity='relu')
    torch.nn.init.constant_(conv.bias, 0)

def init_bn(bn):
    torch.nn.init.constant_(bn.weight, 0.1)

class GConv2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding=True):
        super(GConv2d, self).__init__()

        padding = tuple(map(lambda x:
            (x-1) // 2, kernel_size)) if padding else 0
        self.conv = nn.Conv2d(in_size, out_size,
                kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_size)
        init_conv(self.conv)
        init_bn(self.bn)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

class GConv2d_right(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride):
        super(GConv2d_right, self).__init__()

        self.padding = tuple(map(lambda x: x-1, kernel_size))
        self.padding = [0, self.padding[0], 0, self.padding[1]]
        self.conv = nn.Conv2d(in_size, out_size,
                kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        init_conv(self.conv)
        init_bn(self.bn)

    def forward(self, x):
        return self.bn(F.relu(self.conv(F.pad(x, self.padding))))

class ResNetBlock(nn.Module):
    def __init__(self, io_size, kernel_size, depth):
        super(ResNetBlock, self).__init__()

        self.conv = nn.ModuleList()
        for _ in range(depth):
            self.conv.append(GConv2d(io_size, io_size, kernel_size))

    def forward(self, x):
        h = x.clone()
        for conv in self.conv:
            h = conv(h)
        return x + h

class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super(UpsampleBlock, self).__init__()

        self.pad_size = tuple(map(lambda x: (x - 1) // 2,
                [x for sub in zip(kernel_size, kernel_size)
                    for x in sub]))
        self.conv = GConv2d(in_size, out_size, kernel_size, padding=False)

    def forward(self, x):
        x = F.pad(F.interpolate(x, scale_factor=2), self.pad_size,
                mode='reflect')
        return self.conv(x)

def compute_event_image(events, start, stop, imsize):
    ''' computes event image. Unfortunately on CPU, because of absense of torch.index_max_ function
    '''
    bs = len(start)
    if isinstance(events, torch.Tensor):
        events = events.detach().cpu().numpy()
        start = start.detach().cpu().numpy()
        stop = stop.detach().cpu().numpy()

    x = events[:, 0].astype(int)
    y = events[:, 1].astype(int)
    t = events[:, 2]
    p = events[:, 3].astype(int)
    b = events[:, 4].astype(int)

    # index of the last event for each timestamp
    uniq_b, num_events = np.unique(b, return_counts=True)
    if uniq_b.size < bs:
        # seems, there is an empty set of events along samples
        pass
    shift = np.zeros(bs, dtype=int)
    shift[uniq_b] = np.cumsum(num_events) - 1
    #shift = np.cumsum(num_events) - 1

    tmp = shift[uniq_b]
    assert(np.all(b[tmp] == uniq_b))
    assert(np.all(b[tmp[:-1]+1] > uniq_b[:-1]))
    assert(tmp[-1] + 1 == b.size)

    start_ts = start[b]
    stop_ts = t[shift[b]]

    # normalize timestamps and polarities
    t = (t - start_ts) / np.maximum((stop_ts - start_ts), 1e-9)
    p = (1 - p) // 2 # (-1, 1) -> (1, 0)

    shape = tuple([bs, 4] + list(imsize))
    idx = np.ravel_multi_index([b, p, y, x], shape)
    shift = np.prod([2] + list(imsize))

    res = np.zeros(shape, dtype=np.float32).ravel()
    np.add.at(res, idx, np.ones(idx.size))
    np.maximum.at(res, idx + shift, t)
    return torch.Tensor(res.reshape(shape))

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        in_size = 4
        enc_depth = 4
        tr_depth = 2
        tr_res_depth = 2
        kernel_size = (3, 3)
        # encoder
        sizes = [4]
        self.enc = nn.ModuleList()
        for i in range(enc_depth):
            sizes.append(_BASE_CHANNELS * 2**i)
            self.enc.append(GConv2d_right(sizes[-2], sizes[-1],
                kernel_size, 2))
        # transition
        self.tr = nn.ModuleList()
        for i in range(tr_depth):
            self.tr.append(ResNetBlock(sizes[-1],
                kernel_size=kernel_size, depth=tr_res_depth))
        # decoder
        self.dec = nn.ModuleList()
        self.flow = nn.ModuleList()
        sizes[0] = 32
        for i in range(enc_depth):
            in_size = 2 * sizes[-1-i] + (2 if i>0 else 0)
            out_size = sizes[-2-i]
            self.dec.append(UpsampleBlock(in_size,
                out_size, kernel_size))
            self.flow.append(nn.Conv2d(out_size, 2,
                kernel_size=(1,1)))
            init_conv(self.flow[-1])


    def _get_result(self, flow, outsize):
        return tuple(f[..., :s[0], :s[1]] for f, s in zip(flow, outsize))

    def _extend_size(self, imsize):
        return tuple(map(lambda x: ((x - 1) // 16 + 1) * 16, imsize))

    def forward(self, events, start, stop, imsize, raw=True, intermediate=False):

        # compute extended image size
        outsize = [tuple(map(lambda x: x//2**i, imsize))
                for i in range(len(self.enc))][::-1]

        # compute event_image
        if raw:
            extended_size = self._extend_size(imsize)
            xb = compute_event_image(events, start, stop, extended_size).to(self.device)
        else:
            xb = events

        y = []
        skip = [xb]
        if intermediate:
            intermediate_output = {'input': xb}
        # encoder
        for enc_block in self.enc:
            skip.append(enc_block(skip[-1]))
            if intermediate:
                intermediate_output[f'enc_{len(skip)-2}'] = skip[-1]
        # transition
        h = skip[-1]
        for idx, res in enumerate(self.tr):
            h = res(h)
            if intermediate:
                intermediate_output[f'tr_{idx}'] = h
        # decoder
        n = len(skip)
        for idx, (s, d, f) in enumerate(zip(skip[n:0:-1], self.dec, self.flow)):
            h = torch.cat((h, s), 1)
            if intermediate:
                intermediate_output[f'dec_cat_{idx}'] = h
            h = d(h)
            if intermediate:
                intermediate_output[f'dec_op_{idx}'] = h
            h_flow = f(h)
            if intermediate:
                intermediate_output[f'dec_flow_arth_{idx}'] = h_flow
                y.append(torch.tanh(h_flow).clone()) # clone is required for backward pass
            else:
                y.append(torch.tanh_(h_flow))
            y[-1].mul_(256.)
            h = torch.cat((h, y[-1]), 1)

        # shrink image to original size
        result = self._get_result(y, outsize)
        if intermediate:
            return result, intermediate_output
        return result
