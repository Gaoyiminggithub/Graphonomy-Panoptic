import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from PIL import Image

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

if __name__ == '__main__':
    # Prepare val2017_100 for quick testing:

    dest_dir = os.path.join("./coco/", "annotations/")
    # URL_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"
    # download(URL_PREFIX + "annotations/coco/panoptic_val2017_100.json", dest_dir)
    with open(os.path.join(dest_dir, "panoptic_val2017_100.json")) as f:
        obj = json.load(f)


    def link_val100(dir_full, dir_100):
        print("Creating " + dir_100 + " ...")
        os.makedirs(dir_100, exist_ok=True)
        for img in obj["images"]:
            basename = os.path.splitext(img["file_name"])[0]
            src = os.path.join(dir_full, basename + ".png")
            dst = os.path.join(dir_100, basename + ".png")
            src = os.path.relpath(src, start=dir_100)
            os.symlink(src, dst)


    link_val100(
        os.path.join("./coco", "panoptic_val2017"),
        os.path.join("./coco", "panoptic_val2017_100"),
    )

    link_val100(
        os.path.join("./coco/annotations", "panoptic_val2017_trans"),
        os.path.join("./coco", "panoptic_val2017_trans_100"),
    )