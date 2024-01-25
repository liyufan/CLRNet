import argparse
import datetime
import json
import os

import numpy as np
from lxml import builder, etree
from lxml.etree import _Element as Element
from lxml.etree import _ElementTree as ElementTree
from numpy.typing import NDArray
from tqdm import tqdm


# Reduce the number of points on the same line
def reduce_points(points: NDArray, threshold: int) -> NDArray:
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


def tusimple2voc(save_dir: str, json_file: str, mapping: dict):
    with open(os.path.join(save_dir, json_file), "r") as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    h_samples = list(range(160, 720, 10))

    for i in tqdm(range(len(data))):
        img_name = data[i]["raw_file"]
        base_name = os.path.basename(img_name)

        root, _ = os.path.splitext(base_name)
        xml_file = os.path.join(save_dir, f"{root}.xml")
        out_xml_file = os.path.join(save_dir, f"{root}_label.xml")
        os.system(f"cp {xml_file} {out_xml_file}")

        lanes = data[i]["lanes"]
        categories = data[i]["categories"]
        for i, lane in enumerate(lanes):
            points = []
            for j in range(len(lane)):
                if lane[j] != -2:
                    points.append([lane[j], h_samples[j]])
            points = np.array(points, dtype=np.int32)
            points = reduce_points(points, 10)

            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H-%M-%S")

            add_object_node(out_xml_file, mapping[categories[i]], date, time, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tusimple annotations to voc format"
    )
    parser.add_argument("--save_dir", help="save directory")
    parser.add_argument("--json_file", help="tusimple json file")
    args = parser.parse_args()

    mapping = {
        1: "solid_laneline",
        2: "dashed_laneline",
        3: "dual_solid_laneline",
        4: "solid_dashed_laneline",
    }

    tusimple2voc(args.save_dir, args.json_file, mapping)
