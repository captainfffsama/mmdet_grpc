# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-06-26 14:09:32
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-11-17 16:38:12
@FilePath: /mmdet_grpc/mmdet_grpc/mmdet_ext/__init__.py
@Description:

'''
import mmdet
from packaging.version import Version

if Version(mmdet.__version__) >= Version("3.0.0"):
    from .mmdet3x import extract_feat
else:
    from .mmdet2x import extract_feat

__all__=["extract_feat"]