import argparse
import datetime
import glob
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from lxml import builder, etree
from lxml.etree import _Element as Element
from lxml.etree import _ElementTree as ElementTree
from numpy.typing import NDArray
from tqdm import tqdm

from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.net_utils import load_network
from clrnet.utils.visualization import imshow_lanes


# Reduce the number of points on the same line
def reduce_points(points: NDArray, threshold: int = 10) -> NDArray:
    if points.shape[0] <= 2:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    if x1 == x2:
        return np.array([points[0], points[-1]])
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    dis = np.abs(k * points[:, 0] - points[:, 1] + b) / np.sqrt(k * k + 1)
    index = np.argmax(dis)
    if dis[index] > threshold:
        return np.vstack(
            (
                reduce_points(points[: index + 1], threshold)[:-1],
                reduce_points(points[index:], threshold),
            )
        )
    else:
        return np.array([points[0], points[-1]])


def add_object_node(xml_file: str, name: str, date: str, time: str, points: NDArray):
    dom: ElementTree = etree.parse(
        xml_file, parser=etree.XMLParser(remove_blank_text=True)
    )
    root: Element = dom.getroot()
    maker = builder.ElementMaker()
    points_list = [maker.point(maker.x(str(x)), maker.y(str(y))) for x, y in points]
    object_node = maker.object(
        maker.name(name),
        maker.type("rect"),
        maker.algorithm(
            maker.name("clrnet"),
            maker.date(date),
            maker.time(time),
        ),
        maker.multiPolyline(maker.polyline(*points_list)),
    )
    root.append(object_node)
    dom.write(xml_file, pretty_print=True)


class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)
        self.mapping = {
            1: "solid_laneline",
            2: "dashed_laneline",
            3: "dual_solid_laneline",
            4: "solid_dashed_laneline",
        }

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

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))

        base_name = os.path.basename(data['img_path'])
        root, _ = os.path.splitext(base_name)
        xml_file = os.path.join(self.cfg.savedir, f"{root}.xml")
        if self.cfg.prelabel and not os.path.exists(xml_file):
            raise FileNotFoundError(f"{xml_file} not found")
        out_xml_file = os.path.join(self.cfg.savedir, f"{root}_label.xml")
        if self.cfg.prelabel:
            shutil.copy(xml_file, out_xml_file)

        ori_img = data['ori_img']
        cur_cfg = Config(dict(ori_img_h=ori_img.shape[0], ori_img_w=ori_img.shape[1]))
        cur_cfg.sample_y = range(0, cur_cfg.ori_img_h, 10)

        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H-%M-%S")
        lanes = [lane.to_array(cur_cfg) for lane in data['lanes']]
        categories = [lane.metadata['category'] for lane in data['lanes']]

        if self.cfg.prelabel:
            for lane, category in zip(lanes, categories):
                points = reduce_points(lane.astype(np.int32))
                add_object_node(
                    out_xml_file, self.mapping[category], date, time, points
                )
        if self.cfg.save_img:
            imshow_lanes(
                data['ori_img'],
                lanes,
                categories,
                vis_cls_mapping=self.cfg.vis_cls_mapping,
                show=self.cfg.show,
                out_file=out_file,
            )

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
    cfg.prelabel = args.prelabel
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)


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
    parser.add_argument(
        '--prelabel', action='store_true', help='Whether to start prelabel'
    )
    args = parser.parse_args()
    if args.save_img and args.prelabel:
        raise ValueError('--save_img and --prelabel cannot be set at the same time')
    process(args)
