import torchvision.models as models
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType
from mqbench.utils.state import enable_calibration
from mqbench.utils.state import enable_quantization
from mqbench.convert_deploy import convert_deploy
import torch
import os

# 加载模型
model = models.__dict__["resnet18"](pretrained=True)
model.eval()

# ============================================================
# 自定义 prepare_custom_config_dict
# ============================================================
prepare_custom_config_dict = {
    # 1. extra_qconfig_dict: 自定义量化配置
    #    可以覆盖默认的 qconfig 设置
    'extra_qconfig_dict': {
        # 自定义激活量化配置
        'activation': {
            'fake_quantize': 'learnable_fake_quantize',  # 可选: learnable_fake_quantize, fixed_fake_quantize, dorefa_fake_quantize 等
            'observer': 'ema_minmax',  # 可选: minmax, ema_minmax, mse, ema_mse 等
        },
        # 自定义权重量化配置
        'weight': {
            'fake_quantize': 'learnable_fake_quantize',
            'observer': 'minmax',
            'per_channel': True,  # 是否使用 per-channel 量化
            'symmetry': True,      # 是否使用对称量化
        },
    },

    # 2. extra_quantizer_dict: 量化器的额外参数
    #    控制哪些层需要量化，哪些不需要等
    'extra_quantizer_dict': {
        # 额外需要量化的函数类型（不包含在默认列表中的）
        'additional_function_type': [
            # torch.nn.functional.xxx,
        ],
        # 额外需要量化的模块类型
        'additional_module_type': (
            # torch.nn.LayerNorm,
            # torch.nn.InstanceNorm2d,
        ),
        # 额外需要量化的节点名称
        'additional_node_name': [
            # 'node_name1',
            # 'node_name2',
        ],
        # 排除不量化的模块名称
        'exclude_module_name': [
            # 'fc',  # 不量化 fc 层
            # 'classifier',
        ],
        # 排除不量化的函数类型
        'exclude_function_type': [
            # torch.relu,
            # torch.sigmoid,
        ],
        # 排除不量化的节点名称
        'exclude_node_name': [
            # 'node_name',
        ],
    },

    # 3. preserve_attr: 追踪后需要保留的属性
    #    因为 symbolic_trace 只保存 forward 中的属性
    'preserve_attr': {
        # 格式: {子模块名: [属性列表]}
        '': [],  # 主模块需要保留的属性
        # 'backbone': ['func1', 'func2'],  # backbone 子模块需要保留的属性
        # 'head': ['method1'],
    },

    # 4. concrete_args: 追踪时的具体参数
    #    用于处理 forward 函数中有非 Tensor 参数的情况
    'concrete_args': {
        # 'train': False,  # 如果 forward(self, x, train=False)，可以这样指定
    },

    # 5. extra_fuse_dict: 额外的融合配置
    'extra_fuse_dict': {
        # 额外的融合模式映射
        'additional_fuser_method_mapping': {
            # (torch.nn.Linear, torch.nn.ReLU): torch.nn.intrinsic.LinearReLU,
        },
        # 额外的融合模式
        'additional_fusion_pattern': {
            # ('conv', 'bn'): fusion_function,
        },
        # 额外的 QAT 模块映射
        'additional_qat_module_mapping': {
            # torch.nn.Linear: torch.nn.qat.Linear,
        },
    },

    # 6. leaf_module: 指定不追踪的叶子模块
    #    这些模块会被当作叶子，不进入其内部进行追踪
    'leaf_module': [
        # torch.nn.Identity,
    ],
}

# ============================================================
# 使用自定义配置进行量化
# ============================================================
backend = BackendType.QDQ

# 传入 prepare_custom_config_dict
model = prepare_by_platform(
    model,
    backend,
    prepare_custom_config_dict=prepare_custom_config_dict  # 添加自定义配置
)

print("[INFO] Quantization nodes inserted with custom config.")

# 校准
dummy_inputs = [torch.randn(8, 3, 224, 224) for _ in range(10)]
enable_calibration(model)
with torch.no_grad():
    for dummy_input in dummy_inputs:
        _ = model(dummy_input)

# 量化
enable_quantization(model)

# 导出 ONNX
input_shape = {'data': [1, 3, 224, 224]}
deploy_model = convert_deploy(model, backend, input_shape, model_name='mqbench_qmodel_custom')

onnx_path = 'mqbench_qmodel_custom.onnx'
if os.path.exists(onnx_path):
    print(f"ONNX model exported to {onnx_path}, size: {os.path.getsize(onnx_path)/1024/1024:.2f} MB")