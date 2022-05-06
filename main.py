import os
import argparse
from concurrent import futures

import grpc
from dldetection_pb2_grpc import  add_AiServiceServicer_to_server


from model import MMDetector

from base_config import detector_params,grpc_args

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--cfg", type=str,
                        default="")
    parser.add_argument("-w", "--weight", type=str,
                        default="")
    args = parser.parse_args()
    return args

def main(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_args['max_workers']),
                         options=[('grpc.max_send_message_length',
                                   grpc_args['max_send_message_length']),
                                  ('grpc.max_receive_message_length',
                                   grpc_args['max_receive_message_length'])])
    if os.path.exists(args.cfg) and os.path.exists(args.weight):
        detector_params['cfg_path']=args.cfg_path
        detector_params['ckpt_path']=args.weight
    model=MMDetector(**detector_params)
    add_AiServiceServicer_to_server(model,server)
    server.add_insecure_port("{}:{}".format(grpc_args['host'],grpc_args['port']))
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    args=parse_args()
    main(args)



