# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-03-23 15:49:34
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-11-17 16:38:58
@FilePath: /mmdet_grpc/mmdet_grpc/proto/__init__.py
@Description:
'''
import grpc
from packaging.version import Version


if Version(grpc.__version__) >= Version("1.37.0"):
    from .new import dldetection_pb2_grpc
    from .new import dldetection_pb2
else:
    from .v1370 import dldetection_pb2_grpc
    from .v1370 import dldetection_pb2

__all__=["dldetection_pb2_grpc","dldetection_pb2"]
