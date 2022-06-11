import os
import argparse
from concurrent import futures

import grpc
from dldetection_pb2_grpc import  add_AiServiceServicer_to_server


from model import MMDetector
import base_config as config_manager


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-mc", "--model_cfg", type=str,
                        default="")
    parser.add_argument("-mw", "--model_weight", type=str,
                        default="")
    parser.add_argument("-c", "--cfg", type=str,
                        default="")
    args = parser.parse_args()
    return args

def main(args):
    if os.path.exists(args.cfg):
        config_manager.merge_param(args.cfg)
    args_dict: dict = config_manager.param 
    grpc_args=args_dict['grpc_args']
    detector_params=args_dict['detector_params']
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_args['max_workers']),
                         options=[('grpc.max_send_message_length',
                                   grpc_args['max_send_message_length']),
                                  ('grpc.max_receive_message_length',
                                   grpc_args['max_receive_message_length'])])
    if os.path.exists(args.model_cfg) and os.path.exists(args.model_weight):
        detector_params['cfg_path']=args.model_cfg
        detector_params['ckpt_path']=args.model_weight
    model=MMDetector(**detector_params)
    add_AiServiceServicer_to_server(model,server)
    server.add_insecure_port("{}:{}".format(grpc_args['host'],grpc_args['port']))
    server.start()
    print('mmdet gprc server init done')
    server.wait_for_termination()


if __name__ == "__main__":
    args=parse_args()
    main(args)



