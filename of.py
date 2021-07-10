from .net import Model
import torch
import numpy as np
from os import path as osp

script_dir = osp.dirname(osp.realpath(__file__))


def _expand(value, shape):
    res = np.zeros(shape, value.dtype)
    s = value.shape
    res[:s[0], :s[1], :s[2]] = value
    return res


class OpticalFlow:
    '''
    The OpticalFrom object computes optical flow

    Parameters
    ----------
        imsize (tuple): (height, width) of the resulting optical flow
        model (str): Name of a file with the model parameters.
                     Default: ./data/model/model.pth
        cuda (bool): Indicator of using GPU. Default: True.
    '''
    def __init__(self,
                 imsize,
                 model=osp.join(script_dir, 'data/model/model.pth'),
                 device=torch.device('cuda:0')):

        self._device = torch.device(device)
        if self._device.type == 'cpu':
            self._back = lambda x: x.detach().numpy()
        else:
            self._back = lambda x: x.cpu().detach().numpy()
        self._net = Model(device=self._device)
        self._net.load_state_dict(torch.load(model, map_location=self._device))
        self._net.to(device=self._device)
        self._net.eval()

        self.imsize = imsize

    def __call__(self, events, start, stop, return_all=False):
        ''' Computes optical flow for the input window of events.
        It supports raw events and visualized version of events.

        Parameters
        ----------
            events (tuple): events (x, y, t, p). All entries are iterables.
                            Note: polarities are -1 or 1.
            start (float): a timestamp. A begin of the current window.
            stop (float): a timestamp. An end of the current window.
            return_all (bool): an indicator of returning predictions
                               on every scale. Default: False

        Returns
        -------
            of (np.ndarray): The computed optical flow as 3D
                             tensor with depth 2.
        '''
        with torch.no_grad():
            flow, _, _ = self._net(*self._preprocess(events, start, stop),
                                   self.imsize)
            return self._postprocess(flow, return_all)

    def _collate(self, events, start, stop):
        ''' converts tuple of events for each sample to a single Tensor
            with sample index
        '''
        events = np.hstack([
            np.vstack((
                e,
                np.full_like(e[0], 0),
                np.full_like(e[0], i)))
            for i, e in enumerate(events)
            ])
        timestamps = np.hstack([[b, e] for b, e in zip(start, stop)])
        sample_idx = np.hstack([[i, i] for i in range(len(start))])
        return torch.FloatTensor(events.T, device=self._device), \
               torch.FloatTensor(timestamps, device=self._device), \
               torch.LongTensor(sample_idx, device=self._device)

    def _preprocess(self, events, start, stop):
        return self._collate(events, start, stop)

    def _postprocess(self, flow, return_all):
        def back(flow):
            return np.transpose(self._back(flow), (0, 2, 3, 1))
        if return_all:
            return tuple(map(back, flow))
        return back(flow[-1])
