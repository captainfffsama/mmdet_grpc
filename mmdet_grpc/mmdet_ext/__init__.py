# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-06-26 14:09:32
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-06-26 14:12:47
@FilePath: /mmdet_grpc/mmdet_grpc/mmdet_ext/__init__.py
@Description:

'''
import mmdet

def _version_gt(version_current:str,version_benchmark:str="1.37.0") -> bool:
    for i,j in zip(version_current.split("."),version_benchmark.split(".")):
        if int(i)>=int(j):
            return True

    return False

if _version_gt(mmdet.__version__,"3.0.0"):
    from .mmdet3x import extract_feat
else:
    from .mmdet2x import extract_feat

__all__=["extract_feat"]