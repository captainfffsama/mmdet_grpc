import base64

from collections import defaultdict
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
        req = dldetection_pb2.DlRequest()
        req.imdata = img_b64encode
        response = stub.DlDetection(req)

        req1 = dldetection_pb2.DlEmbeddingRequest()
        req1.imdata = img_b64encode
        req1.imsize.extend((128, 128))
        r1 = stub.DlEmbeddingGet(req1)

        # breakpoint()

        # for obj in response.results:
        #     print("type: {} score: {}  box: {}".format(obj.classid, obj.score,
        #                                                obj.rect.x))

        print(tensor_proto2np(r1).shape)

def test_mmdet(cf,ck):

    change_label: dict = {}

    device: str = 'cuda:0'
    model = init_detector(cf, ck, device=device)
    classes = model.CLASSES if hasattr(
        model, "CLASSES") else model.cfg['METAINFO']['classes']
    label_dict = {
        num: label_name
        if label_name not in change_label else change_label[label_name]
        for num, label_name in enumerate(classes)
    }
    thr=0.3
    if isinstance(thr, float):
        thr = defaultdict(lambda: thr)
    else:
        if 'default' not in thr.keys() and len(thr.keys()) != len(
                label_dict.values()):
            raise ValueError(
                "thr args must be dict or float or have default values")
        else:
            if 'default' not in thr.keys():
                thr = thr
            else:
                default_value = thr.pop('default')
                thr = defaultdict(lambda: default_value)
                thr.update(thr)
    print("model init done!")
    img_path=r"/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/2.jpg"
    img_path1=r"/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/1.jpg"
    breakpoint()
    result = inference_detector(model, [img_path,img_path1])
    print(result)

if __name__ == '__main__':
    run()

    # cf="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/mmdet3x/dino_swin_t_use_pretrained.py"
    # ck="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/mmdet3x/best_31.pth"
    # test_mmdet(cf,ck)

    img_path = r'/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
    # model = init_detector(cf, ck, device="cuda:0")
    # img_path='/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
    # result=extract_feat(model,img_path,img_size=(128,128))
    # print(result.shape)