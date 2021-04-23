# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
import os
import contextlib
import io
import logging

import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

__all__ = ["register_ade_panoptic_separated"]




def my_ADE_dataset_func(name, image_root, sem_seg_root, instances_json):
    panoptic_name = name + "_separated"
    ade_ins_dicts = load_ade_json(instances_json, image_root, panoptic_name)
    ade_sem_dicts = load_sem_seg(sem_seg_root, image_root)
    final_dicts = merge_to_panoptic(ade_ins_dicts, ade_sem_dicts)

    return final_dicts



def load_ade_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes
        # In ADE, we need the id_map to obtain contiguous_id
        # the category ids to contiguous ids in [0, 100].

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map



    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)  # imgs infomation.
    # {'id': 1, 'file_name': '00000001.jpg', 'height': 512, 'width': 683}

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
              "A valid polygon should be a list[float] with even length >= 6."
        )


    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts



# --------------------------------------------------------------
def register_ade_panoptic_separated(
    name, metadata, image_root, panoptic_root, sem_seg_root, instances_json, panoptic_json
):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    It follows the setting used by the PanopticFPN paper:

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name + "_separated"
    # ade_final_dicts = my_ADE_dataset_func(panoptic_name, image_root, sem_seg_root, instances_json)
    DatasetCatalog.register(panoptic_name,
                            lambda: my_ADE_dataset_func(panoptic_name, image_root, sem_seg_root, instances_json))

    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,  # Only in testing
        image_root=image_root,
        panoptic_json=panoptic_json,  # Only in testing
        sem_seg_root=sem_seg_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        **metadata,
    )

    # DatasetCatalog.register(
    #     panoptic_name,
    #     lambda: merge_to_panoptic(
    #         load_coco_json(instances_json, image_root, panoptic_name),
    #         load_sem_seg(sem_seg_root, image_root),
    #     ),
    # )
    # MetadataCatalog.get(panoptic_name).set(
    #     panoptic_root=panoptic_root,
    #     image_root=image_root,
    #     panoptic_json=panoptic_json,
    #     sem_seg_root=sem_seg_root,
    #     json_file=instances_json,  # TODO rename
    #     evaluator_type="coco_panoptic_seg",
    #     ignore_label=255,
    #     **metadata,
    # )
    #
    # semantic_name = name + "_stuffonly"
    # DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
    # MetadataCatalog.get(semantic_name).set(
    #     sem_seg_root=sem_seg_root,
    #     image_root=image_root,
    #     evaluator_type="sem_seg",
    #     ignore_label=255,
    #     **metadata,
    # )


def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results


if __name__ == "__main__":
    """
    Test the COCO panoptic dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco_panoptic \
            path/to/image_root path/to/panoptic_root path/to/panoptic_json dataset_name 10

        "dataset_name" can be "coco_2017_train_panoptic", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image
    import numpy as np

    logger = setup_logger(name=__name__)
    assert sys.argv[4] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[4])

    dicts = load_coco_panoptic_json(sys.argv[3], sys.argv[1], sys.argv[2], meta.as_dict())
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    num_imgs_to_vis = int(sys.argv[5])
    for i, d in enumerate(dicts):
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
        if i + 1 >= num_imgs_to_vis:
            break
