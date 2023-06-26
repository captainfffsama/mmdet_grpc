#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:extract_feat.py
@Author:XuXin
@Date:2023/02/23 19:07:53
@Version:1.0
@Description: 封装后的extract_feat应该接收传入的img参数，输出网络的backbone+neck提取的特征。
作为一个api函数，参考inference_detector, mmdet/apis/inference.py
'''
from typing import Tuple, Optional, Sequence, Union, List

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg


def fix_size(cfg, img_size: Tuple[int, int]) -> dict:

    if isinstance(cfg, list):
        for idx, v in enumerate(cfg):
            fix_size(cfg[idx], img_size)

    elif isinstance(cfg, dict):
        for k in cfg.keys():
            if "scale" == k:
                cfg[k] = img_size
            if "keep_ratio" == k:
                cfg[k] = False
            else:
                if isinstance(cfg[k], list) or isinstance(cfg[k], dict):
                    fix_size(cfg[k], img_size)


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def extract_feat(
    model: nn.Module,
    imgs: ImagesType,
    img_size: Optional[Tuple[int, int]] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(imgs[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

    if img_size is not None:
        fix_size(test_pipeline, img_size)
    test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            data_ = model.data_preprocessor(data_, False)
            embedding = model.extract_feat(data_["inputs"])
            final = embedding[-1].detach().cpu().numpy()

        result_list.append(final)

    if not is_batch:
        return result_list[0]
    else:
        return result_list
