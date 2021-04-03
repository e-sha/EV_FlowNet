import torch
try:
    from torch_scatter import scatter_max, scatter_sum
except:
    scatter_max = None
    scatter_sum = None

def init_conv(conv):
    torch.nn.init.kaiming_normal_(conv.weight, a=4.4, nonlinearity='relu')
    torch.nn.init.constant_(conv.bias, 0)


def init_bn(bn):
    torch.nn.init.constant_(bn.weight, 0.1)


def normalize_timestamps(events, start, stop):
    b = events[:, 4].long()

    start_ts = start[b]
    stop_ts = stop[b]

    # normalize timestamps and polarities
    events[:, 2] = (events[:, 2] - start_ts) / (stop_ts - start_ts)
    return events


def compute_event_image(events, start, stop, imsize, device='cpu', dtype=torch.float32):
    ''' computes event image
    '''
    assert scatter_max is not None, f'follow https://github.com/rusty1s/pytorch_scatter#installation to install torch_scatter'
    bs = len(start)
    if not isinstance(events, torch.Tensor):
        events = torch.tensor(events, device=device)
        start = torch.tensor(start, device=device, dtype=dtype)

    assert len(imsize) == 2

    shape = tuple([bs, 4] + list(imsize))
    res = torch.zeros(shape, dtype=dtype, device=device)
    if events.numel() == 0:
        return res

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].long()
    b = events[:, 4].long()

    assert (torch.abs(p) == 1).all(), f'{torch.unique(p)}'

    # index of the last event for each timestamp
    uniq_b, num_events = torch.unique(b, return_counts=True)
    shift = torch.zeros(bs, dtype=torch.long, device=device)
    shift[uniq_b] = torch.cumsum(num_events, dim=0) - 1

    tmp = shift[uniq_b]
    assert (b[tmp] == uniq_b).all()
    assert (b[tmp[:-1]+1] > uniq_b[:-1]).all()
    assert tmp[-1] + 1 == b.numel()

    dt = torch.zeros(bs, dtype=dtype, device=device)
    dt[uniq_b] = t[shift[uniq_b]] - start[uniq_b]
    dt.clamp_(min=1e-9)

    # normalize timestamps and polarities
    t = (t - start[b]) / dt[b]
    p = (1 - p) // 2 # (-1, 1) -> (1, 0)

    idx = ((b * shape[1] + p) * shape[2] + y) * shape[3] + x

    res = res.view(-1)
    scatter_sum(torch.ones(idx.numel(), device=device, dtype=dtype), idx, out=res)
    scatter_max(t, idx + 2 * imsize[0] * imsize[1], out=res)
    return res.view(*shape)