import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.nn.parameter import Parameter
import pdb
from typing import Dict, List, Optional, Tuple

from .GAT import GAT
import fvcore.nn.weight_init as weight_init

from ..poolers import ROIPooler
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry

__all__ = ["GraphConnection", "GRAPH_CONNECTION_REGISTRY", "build_graph_connection"]
GRAPH_CONNECTION_REGISTRY = Registry("GRAPH_CONNECTION")
GRAPH_CONNECTION_REGISTRY.__doc__ = """
Registry for graph connection, which make graph connection modules.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_graph_connection(cfg, input_shape):
    name = cfg.GRAPH.NAME
    return GRAPH_CONNECTION_REGISTRY.get(name)(cfg, input_shape)


@GRAPH_CONNECTION_REGISTRY.register()
class GraphConnection(nn.Module):
    def __init__(self, cfg,
                 input_shape,):
        super(GraphConnection, self).__init__()
        self.cfg = cfg.clone()
        self.graph_channel = cfg.GRAPH.CHANNEL
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.heads = cfg.GRAPH.HEADS
        self.stuff_out_channel = cfg.GRAPH.STUFF_OUT_CHANNEL
        self.loss_weight_stuff = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.region_in_proj = nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FC_DIM, self.graph_channel)
        self.stuff_in_proj = nn.Linear(cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM * 4, self.graph_channel)
        weight_init.c2_xavier_fill(self.region_in_proj)
        weight_init.c2_xavier_fill(self.stuff_in_proj)

        self.graph = GAT(nfeat=self.graph_channel, nhid=self.graph_channel // self.heads, nclass=self.graph_channel,
                         dropout=0.1, alpha=0.4, nheads=self.heads)

        '''New box head'''
        self.region_out_proj = nn.Linear(self.graph_channel, self.graph_channel)
        weight_init.c2_xavier_fill(self.region_out_proj)

        # in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # box_head = build_box_head(
        #     cfg, ShapeSpec(channels=256, height=pooler_resolution, width=pooler_resolution)
        # )  # TODO: hard code in the channels
        # print(box_head.output_shape)
        box_output_shape = ShapeSpec(channels=cfg.MODEL.ROI_BOX_HEAD.FC_DIM + self.graph_channel)

        self.new_box_predictor = FastRCNNOutputLayers(cfg, box_output_shape)

        '''New mask head'''
        ret_dict = self._init_mask_head(cfg, input_shape)
        self.mask_in_features = ret_dict["mask_in_features"]
        self.new_mask_pooler = ret_dict["mask_pooler"]
        self.new_mask_head = ret_dict["mask_head"]
        # weight_init.c2_xavier_fill(self.new_mask_head)
        '''New segment head'''
        self.stuff_out_proj = nn.Linear(self.graph_channel, self.stuff_out_channel)
        self.seg_score = nn.Conv2d(cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM * 4 + self.stuff_out_channel,
                               cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, 1)
        self.upsample_rate = 4
        weight_init.c2_xavier_fill(self.stuff_out_proj)
        weight_init.c2_xavier_fill(self.seg_score)

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    # def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
    #     """
    #     Forward logic of the mask prediction branch.
    #
    #     Args:
    #         features (dict[str, Tensor]): mapping from feature map names to tensor.
    #             Same as in :meth:`ROIHeads.forward`.
    #         instances (list[Instances]): the per-image instances to train/predict masks.
    #             In training, they can be the proposals.
    #             In inference, they can be the boxes predicted by R-CNN box head.
    #
    #     Returns:
    #         In training, a dict of losses.
    #         In inference, update `instances` with new fields "pred_masks" and return it.
    #     """
    #     if not self.mask_on:
    #         return {} if self.training else instances
    #
    #     if self.training:
    #         # head is only trained on positive proposals.
    #         instances, _ = select_foreground_proposals(instances, self.num_classes)
    #
    #     if self.mask_pooler is not None:
    #         features = [features[f] for f in self.mask_in_features]
    #         boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
    #         features = self.new_mask_pooler(features, boxes)
    #     else:
    #         features = {f: features[f] for f in self.mask_in_features}
    #     return self.new_mask_head(features, instances)

    def forward(self, region_f, proposals, features,
                stuff_f, semseg_score, semseg_targets,
                images=None, seg_result=None, c2d=None, ori_sizes=None, img_ids=None,
                ):
        '''

        Args:
            region_f: region features
            proposals: predicted proposals, containing the gt
            features: fpn features
            stuff_f: stuff features
            semseg_score: predicted sematnic scores
            semseg_targets: semantic segmentation gt
            images: original images
            seg_result:
            c2d:
            ori_sizes:
            img_ids:

        Returns:

        '''
        assert len(proposals) == len(stuff_f)
        bs, _, h, w = semseg_score.shape
        proposals_num = [len(p) for p in proposals]

        assert sum(proposals_num) == len(region_f)

        region_nodes = self.region_in_proj(region_f)
        class_center = torch.matmul(
            F.softmax(semseg_score.flatten(start_dim=2), dim=-1),  # softmax along hw
            stuff_f.flatten(start_dim=2).transpose(1, 2)
        )  # bs x cls x 512
        class_nodes = self.stuff_in_proj(class_center)
        region_nodes_split = region_nodes.split(proposals_num)

        new_region_nodes, new_class_nodes = [], []
        for i in range(bs):
            region_node_per_img = region_nodes_split[i]
            stuff_node_per_img = class_nodes[i]
            nodes_num = len(region_node_per_img) + len(stuff_node_per_img)
            adj = torch.ones(nodes_num, nodes_num).cuda().detach()  # fully connected graph
            graph_nodes = self.graph(torch.cat([region_node_per_img, stuff_node_per_img]), adj)

            new_region_f_per_img, new_stuff_f_per_img = graph_nodes.split(
                [len(region_node_per_img), len(stuff_node_per_img)])
            new_region_nodes.append(new_region_f_per_img)
            new_class_nodes.append(new_stuff_f_per_img)

        new_region_f = torch.cat([region_f, self.region_out_proj(torch.cat(new_region_nodes))], dim=-1)
        new_prediction = self.new_box_predictor(new_region_f)

        # box post-process
        if self.training:
            losses_box = self.new_box_predictor.losses(new_prediction, proposals)
            # losses_mask
            instances, _ = select_foreground_proposals(proposals, self.num_classes)
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.new_mask_pooler(features, boxes)
            losses_mask = self.new_mask_head(features, instances)
        else:
            # testing
            # box
            pred_instances, _ = self.new_box_predictor.inference(new_prediction, proposals)
            # mask
            assert pred_instances[0].has("pred_boxes") and pred_instances[0].has("pred_classes")
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in pred_instances]
            features = self.new_mask_pooler(features, boxes)
            instances = self.new_mask_head(features, pred_instances)

        # stuff
        new_class_nodes = self.stuff_out_proj(torch.stack(new_class_nodes))
        new_stuff_f = torch.matmul(
            F.softmax(semseg_score.flatten(start_dim=2), dim=1).permute(0, 2, 1),  # softmax along nodes
            new_class_nodes
        ).permute(0, 2, 1).view(bs, self.stuff_out_channel, h, w)
        semseg_score = self.seg_score(torch.cat([stuff_f, new_stuff_f], dim=1))
        semseg_score = semseg_score.float()
        segments = F.interpolate(semseg_score, None, self.upsample_rate, mode='bilinear', align_corners=False)
        #
        del semseg_score
        if self.training:
            loss = F.cross_entropy(
                segments, semseg_targets, reduction="mean", ignore_index=self.ignore_value
            )
            losses_sem = {"new_loss_sem_seg": loss * self.loss_weight_stuff}

            # update loss weight name
            # pdb.set_trace()
            losses_box.update(losses_mask)
            losses_box.update(losses_sem)
            key_list = list(losses_box.keys())
            for key in key_list:
                if 'new' not in key:
                    losses_box["new_"+key] = losses_box.pop(key)

            return None, None, losses_box
        else:
            return instances, segments, None

        ''' ###############################   '''
        # if not self.training:
        #     # testing
        #     result = self.post_processor((new_scores, new_bbox_deltas), proposals)
        # else:
        #     loss_classifier, loss_box_reg = self.box_loss_evaluator(
        #         [new_scores], [new_bbox_deltas]
        #     )

        # if self.cfg.GRAPH.MASK_ON:
        #     if not self.training:
        #         x, result, loss_mask = self.mask(features, result, det_targets)
        #     else:
        #         x, result, loss_mask = self.mask(features, proposals, det_targets)

        # new_class_nodes = self.stuff_out_proj(torch.stack(new_class_nodes))
        # new_stuff_f = torch.matmul(
        #     F.softmax(semseg_score.flatten(start_dim=2), dim=1).permute(0, 2, 1),  # softmax along nodes
        #     new_class_nodes
        # ).permute(0, 2, 1).view(bs, self.stuff_out_channel, h, w)
        #
        # semseg_score = self.score(torch.cat([stuff_f, new_stuff_f], dim=1))
        # if self.upsample_rate != 1:
        #     segments = F.interpolate(semseg_score, None, self.upsample_rate, mode='bilinear', align_corners=True)
        #
        # new_segment_loss = self.segment_loss(segments, semseg_targets, images, seg_result=seg_result, c2d=c2d,
        #                                      ori_sizes=ori_sizes, img_ids=img_ids)

        # if self.training:
        #     return None, {
        #         'loss_classifier_bgr': loss_classifier / 2,
        #         'loss_box_reg_bgr': loss_box_reg / 2,
        #         'loss_mask_bgr': loss_mask["loss_mask"] / 2,
        #         'loss_segmentation_bgr': new_segment_loss / 2,
        #     }
        # else:
        #     return result, None

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks