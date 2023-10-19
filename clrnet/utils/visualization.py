import cv2
import os
import os.path as osp
import numpy as np
from cv2.typing import MatLike


def put_legend(img: MatLike, mapping: dict):
    pos = 40
    for i in range(len(mapping)):
        cv2.putText(
            img,
            mapping[i + 1]['name'],
            (0, pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mapping[i + 1]['color'],
            2,
            cv2.LINE_AA,
        )
        pos += 30


def imshow_lanes(img, lanes, categories, vis_cls_mapping, show=False, out_file=None, width=4):
    lanes_xys = []
    lanes_categories = []
    for i, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        if xys:
            lanes_xys.append(xys)
            lanes_categories.append(categories[i])
    if lanes_xys:
        _zip = zip(lanes_xys, lanes_categories, strict=True)
        lanes_xys, lanes_categories = zip(*sorted(_zip, key=lambda e: e[0][0][0]))

    res_img = img.copy()
    put_legend(res_img, vis_cls_mapping)

    for idx, xys in enumerate(lanes_xys):
        color = vis_cls_mapping[lanes_categories[idx]]['color']
        for i in range(1, len(xys)):
            cv2.line(res_img, xys[i - 1], xys[i], color, thickness=width)

    res_img = np.hstack((img, res_img))

    if show:
        cv2.imshow('view', res_img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, res_img)
