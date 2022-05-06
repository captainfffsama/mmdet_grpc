import base64
import numpy as np
import cv2

from mmdet.apis import init_detector, inference_detector
import dldetection_pb2
from dldetection_pb2_grpc import AiServiceServicer


class MMDetector(AiServiceServicer):
    def __init__(self, cfg_path, ckpt_path, thr, change_label={}):
        self.model = init_detector(cfg_path, ckpt_path, device="cuda:0")
        self.label_dict = {
            num: label_name
            if label_name not in change_label
            else change_label[label_name]
            for num, label_name in enumerate(self.model.CLASSES)
        }
        self.thr = thr
        print("model init done!")

    def _standardized_result(self, result) -> list:
        new_result = []
        for idx, objs_info_matrix in enumerate(result):
            if objs_info_matrix.shape[0] > 0:
                new_result += [
                    (self.label_dict[idx], obj[-1], *obj[:-1])
                    for obj in objs_info_matrix
                ]
        return new_result

    def _filter_obj_by_thr(self, result):
        r"""按照阈值过滤目标，并将坐标整数化"""
        after_filter_result = []
        for obj in result:
            label = obj[0]
            score = obj[1]
            if score >= self.thr:
                after_filter_result.append((label, score, *[int(i) for i in obj[-4:]]))
        return after_filter_result

    def infer(self, img):
        result = inference_detector(self.model, img)
        new_result = self._standardized_result(result)
        new_result=self._filter_obj_by_thr(new_result)
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


        return result_pro
