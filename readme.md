[toc]

# 注意

针对 mmdet进行了适应,理论上还兼容2x系列,但是2x系列没有测试

# 依赖

numpy
opencv-python-headless
torch >1.6
torchvision
mmdet >=3.0.0
protobuf >= 3.15.2
grpcio >= 1.37.0
grpcio-tools >= 1.37.0
pid
pyyaml
simecy @ git+https://github.com/captainfffsama/simple_encryption.git@v0.2#egg=simecy

# 文件加密密码

`chiebot-ai`

# 使用

`base_config.py` 中可以配置目标检测和grpc的一些参数

在 `main.py`中也可以添加目标检测的配置文件和权重路径
