from pathlib import Path
from imageio import imwrite
from skimage.color import hsv2rgb
import numpy as np
from tqdm import tqdm

import sys


cur_path = Path(__file__).parent.resolve()
module_name = cur_path.name
sys.path.append(str(cur_path.parent))
OpticalFlow = __import__(module_name).OpticalFlow

data_base = cur_path/'data'/'events'
out_path = cur_path/'res'
out_path.mkdir(parents=True, exist_ok=True)

# events. Note that polarity values are in {-1, +1}
events = np.load(str(data_base/'dvs0.npy'))
# number of frames per second
fps = 120
# height and width of images
imsize = 480, 640
# window size in microseconds
dt = 1. / fps


def vis_flow(flow):
    mag = np.linalg.norm(flow, axis=2)
    a_mag = np.min(mag)
    b_mag = np.max(mag)

    ang = np.arctan2(flow[..., 0], flow[..., 1])
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[:2]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) *
                    (255. / (b_mag - a_mag + 1e-32))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)


def vis_events(events, imsize):
    res = np.zeros(imsize, dtype=np.uint8).ravel()
    x, y = map(lambda x: x.astype(int), events[:2])
    i = np.ravel_multi_index([y, x], imsize)
    np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))


def collage(flow_rgb, events_rgb):
    flow_rgb = flow_rgb[::-1]

    orig_h, orig_w, c = flow_rgb[0].shape
    h = orig_h + flow_rgb[1].shape[0]
    w = orig_w + events_rgb.shape[1]

    res = np.zeros((h, w, c), dtype=events_rgb.dtype)
    res[:orig_h, :orig_w] = flow_rgb[0]
    res[:orig_h, orig_w:] = events_rgb

    left = 0
    for img in flow_rgb[1:]:
        h, w = img.shape[:2]
        right = left + w
        res[orig_h:orig_h+h, left:right] = img
        left = right
    return res


of = OpticalFlow(imsize)

x, y, t, p = events

start_t = t[0]
stop_t = t[-1]
frame_ts = np.arange(start_t, stop_t, dt)
frame_ts = np.append(frame_ts, [frame_ts[-1] + dt])
num_frames = len(frame_ts) - 1

idx_array = np.searchsorted(t, frame_ts)
for i, (b, e) in tqdm(enumerate(zip(idx_array[:-1],
                                    idx_array[1:])),
                      total=num_frames):
    # events of the current sliding window
    frame_events = [x[b:e] for x in events]
    # predicted optical flow. Batch size is equal to 1
    flow = of([frame_events], [frame_ts[i]], [frame_ts[i+1]], return_all=True)
    flow = tuple(map(np.squeeze, flow))
    # visualization
    events_rgb = vis_events(frame_events, imsize)
    flow_rgb = list(map(vis_flow, flow))
    imwrite(str(out_path/'{:04d}.jpg'.format(i+1)),
            collage(flow_rgb, events_rgb))
