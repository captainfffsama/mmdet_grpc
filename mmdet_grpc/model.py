import base64
from collections import defaultdict
from typing import Union
from simecy import decrypt
import torch

import numpy as np
import cv2

import mmdet
from mmdet.apis import init_detector, inference_detector
from mmdet_grpc.proto import dldetection_pb2
from mmdet_grpc.proto import dldetection_pb2_grpc as dld_pb2_grpc

from mmdet_grpc.mmdet_ext import extract_feat
from .utils import np2tensor_proto, version_gt


class MMDetector(dld_pb2_grpc.AiServiceServicer):

    def __init__(self,
                 cfg_path,
                 ckpt_path,
                 thr: Union[float, dict],
                 change_label: dict = {},
                 device: str = 'cuda:0'):
        with decrypt(cfg_path,
                     'chiebot-ai') as cf, decrypt(ckpt_path,
                                                  'chiebot-ai') as ck:
            self.model = init_detector(cf, ck, device=device)
        classes = self.model.CLASSES if hasattr(
            self.model, "CLASSES") else self.model.cfg['METAINFO']['classes']
        self.label_dict = {
            num: label_name
            if label_name not in change_label else change_label[label_name]
            for num, label_name in enumerate(classes)
        }
        if isinstance(thr, float):
            self.thr = defaultdict(lambda: thr)
        else:
            if 'default' not in thr.keys() and len(thr.keys()) != len(
                    self.label_dict.values()):
                raise ValueError(
                    "thr args must be dict or float or have default values")
            else:
                if 'default' not in thr.keys():
                    self.thr = thr
                else:
                    default_value = thr.pop('default')
                    self.thr = defaultdict(lambda: default_value)
                    self.thr.update(thr)
        print("model init done!")

    def _mmdet2x_standardized_result(self, result) -> list:
        new_result = []
        for idx, objs_info_matrix in enumerate(result):
            if objs_info_matrix.shape[0] > 0:
                new_result += [(self.label_dict[idx], obj[-1], *obj[:-1])
                               for obj in objs_info_matrix]
        return new_result

    def _mmdet3x_standardized_result(self, result) -> list:
        from mmdet.structures import DetDataSample, SampleList
        new_result = []
        result:DetDataSample=result
        scores = result.pred_instances.scores.detach().cpu().numpy().tolist()
        labels = result.pred_instances.labels.detach().cpu().numpy().tolist()
        bboxes=result.pred_instances.bboxes.detach().cpu().numpy()
        for s,l,box in zip(scores,labels,bboxes):
            new_result.append((self.label_dict[l],s,*(box.tolist())))

        return new_result


    def _standardized_result(self, result) -> list:
        if not version_gt(mmdet.__version__, "3.0.0"):
            return self._mmdet2x_standardized_result(result)
        else:
            return self._mmdet3x_standardized_result(result)

    def _filter_obj_by_thr(self, result):
        r"""按照阈值过滤目标，并将坐标整数化"""
        after_filter_result = []
        for obj in result:
            label = obj[0]
            score = obj[1]
            if score >= self.thr[label]:
                after_filter_result.append(
                    (label, score, *[int(i) for i in obj[-4:]]))
        return after_filter_result

    def infer(self, img):
        result = inference_detector(self.model, img)
        new_result = self._standardized_result(result)
        new_result = self._filter_obj_by_thr(new_result)
        return new_result

    def DlDetection(self, request, context):
        img_base64 = base64.b64decode(request.imdata)

        img_array = np.fromstring(img_base64, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        result = self.infer(img)
        print(result)
        result_pro = dldetection_pb2.DlResponse()
        for obj in result:
            obj_pro = result_pro.results.add()
            obj_pro.classid = obj[0]
            obj_pro.score = float(obj[1])
            obj_pro.rect.x = int(obj[2])
            obj_pro.rect.y = int(obj[3])
            obj_pro.rect.w = int(obj[4] - obj[2])
            obj_pro.rect.h = int(obj[5] - obj[3])
        torch.cuda.empty_cache()
        return result_pro

    def DlEmbeddingGet(self, request, conext):
        img_base64 = base64.b64decode(request.imdata)
        img_array = np.fromstring(img_base64, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

        im_size = tuple(request.imsize)

        result: np.ndarray = extract_feat(self.model, img, img_size=im_size)
        print("embedding size: ", result.shape)
        torch.cuda.empty_cache()
        return np2tensor_proto(result)