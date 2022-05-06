detector_params = dict(
    cfg_path="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/detr.py",
    ckpt_path="/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/best.pth",
    thr=0.3,
    change_label=dict(wcgz_dxdk="wcgz"),
)

grpc_args = dict(
    host="127.0.0.1",
    port="7999",
    max_workers=1,
    max_send_message_length=10 * 1024 * 1024,
    max_receive_message_length=10 * 1024 * 1024,
)
