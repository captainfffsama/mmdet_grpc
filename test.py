import base64

import grpc
from mmdet_grpc.proto import dldetection_pb2

from mmdet_grpc.proto import dldetection_pb2_grpc as dld_grpc
from mmdet_grpc.utils import tensor_proto2np

from mmdet_grpc.mmdet_ext import extract_feat
from mmdet.apis import init_detector, inference_detector
import cv2


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:52007') as channel:
        stub = dld_grpc.AiServiceStub(channel)
        img_path = r'/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
        img_path=r"/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/2.jpg"
        img_file = open(img_path,'rb')  # 二进制打开图片文件
        img_b64encode = base64.b64encode(img_file.read())  # base64编码
        img_file.close()  # 文件关闭
        # req = dldetection_pb2.DlRequest()
        # req.imdata = img_b64encode
        # response = stub.DlDetection(req)

        req1 = dldetection_pb2.DlEmbeddingRequest()
        req1.imdata = img_b64encode
        req1.imsize.extend((128, 128))
        r1 = stub.DlEmbeddingGet(req1)

        # breakpoint()

        # for obj in response.results:
        #     print("type: {} score: {}  box: {}".format(obj.classid, obj.score,
        #                                                obj.rect.x))

        print(tensor_proto2np(r1).shape)


if __name__ == '__main__':
    run()

    # cf="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/detr.py"
    # ck="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/best.pth"

    # model = init_detector(cf, ck, device="cuda:0")
    # img_path='/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
    # result=extract_feat(model,img_path,img_size=(128,128))
    # print(result.shape)