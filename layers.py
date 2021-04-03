import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path

from .helpers import init_conv, init_bn, normalize_timestamps, compute_event_image


class GConv2d(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride=1,
                 padding=True,
                 activation=nn.ReLU(),
                 use_bn=True):
        super(GConv2d, self).__init__()

        padding = tuple(map(lambda x: (x-1) // 2,
                            kernel_size)) if padding else 0
        self.conv = nn.Conv2d(in_size,
                              out_size,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding)
        init_conv(self.conv)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_size)
            init_bn(self.bn)
        self.activation = activation

    def forward(self, x):
        forwarded = self.activation(self.conv(x))
        if hasattr(self, 'bn'):
            return self.bn(forwarded)
        else:
            return forwarded


class GConv2d_right(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 activation=nn.ReLU()):
        super(GConv2d_right, self).__init__()

        self.padding = tuple(map(lambda x: x-1, kernel_size))
        self.padding = [0, self.padding[0], 0, self.padding[1]]
        self.conv = nn.Conv2d(in_size,
                              out_size,
                              kernel_size=kernel_size,
                              stride=stride)
        init_conv(self.conv)
        self.bn = nn.BatchNorm2d(out_size)
        init_bn(self.bn)
        self.activation = activation

    def forward(self, x):
        forwarded = self.activation(self.conv(F.pad(x, self.padding)))
        return self.bn(forwarded)


class ResNetBlock(nn.Module):
    def __init__(self, io_size, kernel_size, depth, activation):
        super(ResNetBlock, self).__init__()

        self.conv = nn.ModuleList()
        for _ in range(depth):
            self.conv.append(
                GConv2d(
                    in_size=io_size,
                    out_size=io_size,
                    kernel_size=kernel_size,
                    activation=activation
                )
            )

    def forward(self, x):
        h = x.clone()
        for conv in self.conv:
            h = conv(h)
        return x + h


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(UpsampleBlock, self).__init__()

        self.pad_size = tuple(map(lambda x: (x - 1) // 2,
                                  [x for sub in zip(kernel_size,
                                                    kernel_size)
                                   for x in sub]))
        self.conv = GConv2d(in_size,
                            out_size,
                            kernel_size,
                            padding=False,
                            activation=activation)

    def forward(self, x):
        x = F.pad(F.interpolate(x, scale_factor=2),
                  self.pad_size,
                  mode='reflect')
        return self.conv(x)


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must " \
                                    "have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must " \
                                   "have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = (Path(__file__).resolve().parent /
                "quantization_layer_init" /
                "trilinear_init.pth")
        if path.is_file():
            state_dict = torch.load(str(path), map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm(range(1000), desc='ValueLayer::init_kernel'):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim)
        self.dim = dim

    def forward(self, events, shape, B):
        dtype = events.dtype
        device = events.device
        dim = [self.dim] + list(shape)
        # points is a list, since events can have any size
        num_voxels = int(2 * np.prod(dim) * B)
        if events.numel() == 0:
            return torch.zeros([B, 2 * dim[0]] + dim[1:],
                               dtype=dtype,
                               device=device)
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        C, H, W = dim

        # get values for each channel
        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3]
        b = events[:, 4]

        with torch.no_grad():
            p = (p + 1) / 2  # map polarity to 0, 1

            idx_before_bins = (x.long()
                               + W * y.long()
                               + 0
                               + W * H * C * p.long()
                               + W * H * C * 2 * b.long())

        if False:
            with torch.no_grad():
                cells = torch.arange(C,
                                     dtype=torch.long,
                                     device=device).view(1, -1)
                idx = idx_before_bins.view(-1, 1) + W * H * cells
                t = t.view(-1, 1)
            values = t * self.value_layer.forward(t - cells.to(dtype) / (C-1))
            vox.put_(idx, values, accumulate=True)
        else:
            for i_bin in range(C):
                values = t * self.value_layer.forward(t-i_bin/(C-1))

                # draw in voxel grid
                idx = idx_before_bins + W * H * i_bin
                vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class InverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output, None


class InverseGradientLayer(nn.Module):
    def forward(self, input_):
        return InverseGradient.apply(input_)

