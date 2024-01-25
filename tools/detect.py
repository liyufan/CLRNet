import argparse
import glob
import json
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.net_utils import load_network
from clrnet.utils.visualization import imshow_lanes


class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height :, :, :].astype(np.float32)
        data = {'img': img, 'lanes': [], 'categories': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': img_path, 'ori_img': ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def convert_lane(self, lane):
        ys = np.array(list(range(160, 720, 10))) / self.cfg.ori_img_h
        xs = lane(ys)
        invalid_mask = xs < 0
        lane = (xs * self.cfg.ori_img_w).astype(int)
        lane[invalid_mask] = -2
        return lane.tolist()

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        categories = [lane.metadata['category'] for lane in data['lanes']]
        if self.cfg.save_img:
            imshow_lanes(
                data['ori_img'],
                lanes,
                categories,
                vis_cls_mapping=self.cfg.vis_cls_mapping,
                show=self.cfg.show,
                out_file=out_file,
            )
        pred = {
            'raw_file': osp.basename(data['img_path']),
            'lanes': [self.convert_lane(lane) for lane in data['lanes']],
            'categories': categories,
        }
        data['pred'] = json.dumps(pred)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data


def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        exts = ['jpg', 'png', 'jpeg']
        if sys.platform != 'win32':
            exts += [e.upper() for e in exts]
        paths = []
        for ext in exts:
            paths += sorted(glob.glob(os.path.join(p, f'*.{ext}')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths


def process(args):
    cfg = Config.fromfile(args.config)
    cfg.save_img = args.save_img
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    lines = []
    for p in tqdm(paths):
        lines.append(detect.run(p)['pred'])
    if not cfg.save_img:
        os.makedirs(cfg.savedir, exist_ok=True)
    with open(osp.join(cfg.savedir, 'pred.json'), 'w') as output_file:
        output_file.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument(
        '--img',
        help='The path of the img (img file or img_folder), for example: "data/*.png"',
    )
    parser.add_argument(
        '--save_img', action='store_true', help='Whether to save the image'
    )
    parser.add_argument('--show', action='store_true', help='Whether to show the image')
    parser.add_argument(
        '--savedir', type=str, default=None, help='The root of save directory'
    )
    parser.add_argument(
        '--load_from', type=str, default='best.pth', help='The path of model'
    )
    args = parser.parse_args()
    process(args)
