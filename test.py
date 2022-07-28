import base64

import grpc
import dldetection_pb2
from dldetection_pb2_grpc import AiServiceStub

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:52007') as channel:
        stub =AiServiceStub (channel)
        img_file = open(r'/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg','rb')   # 二进制打开图片文件
        img_b64encode = base64.b64encode(img_file.read())  # base64编码
        img_file.close()  # 文件关闭
        req=dldetection_pb2.DlRequest()
        req.imdata=img_b64encode
        response = stub.DlDetection(req)

        # breakpoint()

        for obj in response.results:
            print("type: {} score: {}".format(obj.classid,obj.score))


if __name__ == '__main__':
    run()
