from argparse import ArgumentParser
from imageio import imwrite
import numpy as np
from pathlib import Path
from PIL import ImageDraw, Image
from skimage.color import hsv2rgb
import sys
import torch
from tqdm import tqdm


cur_path = Path(__file__).parent.resolve()


def parse_args():
    parser = ArgumentParser(description="Applies the network "
                                        "to the sample data")
    parser.add_argument('-m', '--model',
                        help='Model to apply',
                        type=Path,
                        default=cur_path/'data/model/model.pth',
                        required=False)
    parser.add_argument('--fps',
                        help='Frames per second to reconstruct',
                        type=int,
                        default=120,
                        required=False)
    parser.add_argument('--width',
                        help='Image width',
                        type=int,
                        default=640,
                        required=False)
    parser.add_argument('--height',
                        help='Image height',
                        type=int,
                        default=480,
                        required=False)
    parser.add_argument('--device',
                        help='Device to use by torch',
                        type=torch.device,
                        default='cpu',
                        required=False)
    return parser.parse_args()


def euclidian2polar(data):
    assert data.shape[-1] == 2
    magnitude = np.linalg.norm(data, axis=data.ndim-1)
    angle = np.arctan2(data[..., 0], data[..., 1]) + np.pi
    return magnitude, angle


def vis_flow(flow):
    mag, ang = euclidian2polar(flow)
    a_mag = np.min(mag)
    b_mag = max(np.max(mag), 6)

    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[:2]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) *
                    (255. / (b_mag - a_mag + 1e-4))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)


def vis_events(events, imsize):
    res = np.zeros(imsize, dtype=np.uint8).ravel()
    x, y = map(lambda x: x.astype(int), events[:2])
    i = np.ravel_multi_index([y, x], imsize)
    np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))


def collage(flow_rgb, events_rgb, text=None):
    flow_rgb = flow_rgb[::-1]

    orig_h, orig_w, c = flow_rgb[0].shape
    text_h = 15
    h = text_h + orig_h + flow_rgb[1].shape[0]
    w = orig_w + events_rgb.shape[1]

    res = np.zeros((h, w, c), dtype=events_rgb.dtype)
    res[text_h:text_h + orig_h, :orig_w] = flow_rgb[0]
    res[text_h:text_h + orig_h, orig_w:] = events_rgb

    left = 0
    for img in flow_rgb[1:]:
        h, w = img.shape[:2]
        right = left + w
        res[text_h + orig_h:text_h+orig_h+h, left:right] = img
        left = right

    if text is None:
        return res
    assert isinstance(text, str)
    image = Image.fromarray(res)
    ImageDraw.Draw(image).text((0, 0), text, (255, 255, 255))
    return np.asarray(image)


def visualize(events, flow):
    imsize = flow[-1].shape[:2]
    magnitude, _ = euclidian2polar(flow[-1])
    text = f'Magnitude: mean={np.mean(magnitude):.2f}, ' \
           f'median={np.median(magnitude):.2f}, ' \
           f'min={np.min(magnitude):.2f}, ' \
           f'max={np.max(magnitude):.2f}'
    events_rgb = vis_events(events, imsize)
    flow_rgb = list(map(vis_flow, flow))
    return collage(flow_rgb, events_rgb, text)


def init_model(args):
    module_name = cur_path.name
    sys.path.append(str(cur_path.parent))
    return __import__(module_name).OpticalFlow((args.height, args.width),
                                               model=args.model,
                                               device=args.device)


def get_event_iterator(args):
    # events. Note that polarity values are in {-1, +1}
    events = np.load(str(cur_path/'data'/'events'/'dvs0.npy'))
    x, y, t, p = events
    start_t = t[0]
    stop_t = t[-1]

    dt = 1. / args.fps
    frame_ts = np.arange(start_t, stop_t + dt, dt)
    num_frames = len(frame_ts) - 1

    idx_array = np.searchsorted(t, frame_ts)
    num_frames = len(frame_ts) - 1

    def frame_iter():
        for b, e, start_ts, stop_ts in zip(idx_array[:-1], idx_array[1:],
                                           frame_ts[:-1], frame_ts[1:]):
            yield [x[b:e] for x in events], start_ts, stop_ts
    return num_frames, frame_iter()


def main(args):
    global device
    device = args.device
    out_path = cur_path/'res'
    out_path.mkdir(parents=True, exist_ok=True)

    of = init_model(args)
    num_frames, frame_iter = get_event_iterator(args)
    for i, (frame_events, start_ts, stop_ts) in tqdm(enumerate(frame_iter),
                                                     total=num_frames):
        # predicted optical flow. Batch size is equal to 1
        flow = of([frame_events], [start_ts], [stop_ts], return_all=True)
        flow = tuple(map(np.squeeze, flow))
        # flow = [np.zeros([480//2**i, 640//2**i, 2]) for i in range(4)[::-1]]
        # visualization
        imwrite(str(out_path/'{:04d}.jpg'.format(i+1)),
                visualize(frame_events, flow))


if __name__ == '__main__':
    args = parse_args()
    main(args)
