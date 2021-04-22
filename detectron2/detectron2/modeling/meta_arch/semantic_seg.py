# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.layers import DeformConv, ModulatedDeformConv
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import Backbone, build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

import pdb

__all__ = ["SemanticSegmentor", "SEM_SEG_HEADS_REGISTRY", "SemSegFPNHead", "build_sem_seg_head", "SemSegFPNHead_DFConv"]


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float]
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        self.scale_heads = []
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

#### New module
@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead_DFConv(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
        num_layers: int = 3,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        '''
        Parameters check
        '''
        # assert len(set(self.in_features)) == 1
        # assert len(set(feature_channels)) == 1
        in_channels = feature_channels[0]
        # Deformable conv
        self.subnet = FCN_DFConv(in_channels=in_channels,
                                 out_channels=conv_dims,
                                 num_layers=num_layers)

        self.predictor = Conv2d(conv_dims*len(self.in_features), num_classes, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.predictor.weight.data, 0, 0.01)
        self.predictor.bias.data.zero_()
        # weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "num_layers": cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features['p2'], features['p3'], features['p4'], features['p5']
        fpn_p2 = self.subnet(fpn_p2)
        fpn_p3 = self.subnet(fpn_p3)
        fpn_p4 = self.subnet(fpn_p4)
        fpn_p5 = self.subnet(fpn_p5)

        fpn_p3 = F.interpolate(fpn_p3, None, 2, mode='bilinear', align_corners=True)
        fpn_p4 = F.interpolate(fpn_p4, None, 4, mode='bilinear', align_corners=True)
        fpn_p5 = F.interpolate(fpn_p5, None, 8, mode='bilinear', align_corners=True)

        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)
        x = self.predictor(feat)
        return x

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead_DFConvwGraph(SemSegFPNHead_DFConv):

    def forward(self, features, targets=None):
        """
                Returns:
                    In training, returns (None, dict of losses, features, predicted logits with small size)
                    In inference, returns (CxHxW logits, {}, features, predicted logits with small size)
                """
        x, feat = self.layers(features)
        if self.training:
            return None, self.losses(x, targets), feat, x
        else:
            x_upsample = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x_upsample, {}, feat, x

    def layers(self, features):
        fpn_p2, fpn_p3, fpn_p4, fpn_p5 = features['p2'], features['p3'], features['p4'], features['p5']
        fpn_p2 = self.subnet(fpn_p2)
        fpn_p3 = self.subnet(fpn_p3)
        fpn_p4 = self.subnet(fpn_p4)
        fpn_p5 = self.subnet(fpn_p5)

        fpn_p3 = F.interpolate(fpn_p3, None, 2, mode='bilinear', align_corners=True)
        fpn_p4 = F.interpolate(fpn_p4, None, 4, mode='bilinear', align_corners=True)
        fpn_p5 = F.interpolate(fpn_p5, None, 8, mode='bilinear', align_corners=True)

        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)
        x = self.predictor(feat)
        return x, feat


class FCN_DFConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1):
        super(FCN_DFConv, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers

        assert padding == dilation * (kernel_size - 1) // 2

        # offset_base_channels = kernel_size * kernel_size
        # offset_channels = offset_base_channels * 3 #default: 27
        # self.offset_conv =

        conv_item = []
        for i in range(num_layers):
            if i == num_layers - 2:
                dfconv = DFConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                     stride=stride, dilation=dilation)
                # weight_init.c2_msra_fill(dfconv)
                in_channels = out_channels
            else:
                dfconv = DFConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                                     stride=stride, dilation=dilation)
            conv_item.append(dfconv)
            conv_item.append(nn.ReLU(True))
        self.conv = nn.Sequential(*conv_item)

    def forward(self, x):
        x = self.conv(x)
        return x

# class DFConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size: int = 3, stride: int = 1, padding: int = 1,
#                  dilation: int = 1, deformable_groups: int = 1,
#                  groups: int = 1):
#         super(DFConvLayer, self).__init__()
#         assert padding == dilation * (kernel_size - 1) // 2
#
#         offset_base_channels = kernel_size * kernel_size
#         offset_channels = offset_base_channels * 3 #default: 27
#         self.offset = Conv2d(
#             in_channels,
#             deformable_groups * offset_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=groups,
#             dilation=dilation
#         )
#         weight_init.c2_msra_fill(self.offset)
#
#         self.dfconv = ModulatedDeformConv(in_channels, out_channels, kernel_size=kernel_size,
#                                                      stride=stride, padding=padding, dilation=dilation)
#
#     def forward(self, x):
#         if x.numel() == 0:
#             return self.dfconv(x, None, None)
#         offset_mask = self.offset(x)
#         offset = offset_mask[:, :18, :, :]
#         mask = offset_mask[:, -9:, :, :].sigmoid()
#         x = self.dfconv(x, offset, mask)
#         return x


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""
    def __init__(
        self,
        in_channels,
        out_channels,
        with_modulated_dcn=True,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
        bias=False
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2
            )
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3 #default: 27
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2 #default: 18
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        for l in [self.offset,]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.)
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride
            )
        ]
        from detectron2.layers.wrappers import _NewEmptyTensorOp
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
