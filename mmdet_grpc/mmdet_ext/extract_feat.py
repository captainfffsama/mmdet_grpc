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
from typing import Tuple, Optional

import numpy as np
from copy import deepcopy
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.utils import replace_ImageToTensor


def fix_size(cfg, img_size: Tuple[int, int]) -> dict:

    if isinstance(cfg, list):
        for idx, v in enumerate(cfg):
            fix_size(cfg[idx], img_size)

    elif isinstance(cfg, dict):
        for k in cfg.keys():
            if "img_scale" == k:
                cfg[k] = img_size
            else:
                if isinstance(cfg[k], list) or isinstance(cfg[k], dict):
                    fix_size(cfg[k], img_size)


def extract_feat(model,
                 imgs,
                 batch_size=1,
                 img_size: Optional[Tuple[int, int]] = None):
    """提取图片的feature embedding。支持单GPU，batch size > 1。

    Args:
        model (nn.Module): 检测器模型.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]): 可以是图片列表或读入的文件。
        batch_size (int, optional): batch size. 默认为1.
        img_size (Tuple[int,int],Optional): 默认为None 不修改test pipeline中分辨率

    Returns:
        _type_: 如果imgs为list或tuple，返回长度相同的list。否则直接返回单个图片feat。
        feat的shape为：
        [batch_size, out_channels, imgh/64, imgw/64]
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = deepcopy(model.cfg)
    device = next(model.parameters()).device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    if img_size is not None:
        fix_size(cfg.data.test.pipeline,img_size)
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            data = dict(img=img)
        else:
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # go through the pipeline
        data = test_pipeline(data)
        datas.append(data)

    # add batchsize, 3d-> 4d
    len_all = len(imgs)
    batch_num = len_all // batch_size
    if len_all % batch_size != 0:
        batch_num += 1
    all_batch_datas = []

    for bi in range(batch_num):
        # torch.empty()
        batch_datas = []
        for i in range(bi * batch_size, (bi + 1) * batch_size):
            if i < len_all:
                batch_datas.append(datas[i])

        data = collate(batch_datas, samples_per_gpu=batch_size)
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'
        all_batch_datas.append(data)
    # forward the model
    feat_all_data = []
    with torch.no_grad():
        for data in all_batch_datas:
            img = data['img']
            # with batch_size
            feat: Tuple[torch.Tensor] = model.extract_feat(img[0])
            feat_all_data.append(feat[-1].detach().cpu().numpy())
    return feat_all_data if len_all > 1 else feat_all_data[0]