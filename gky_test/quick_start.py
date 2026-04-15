import torchvision.models as models                           # for example model
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy
import os                                                     # for file operations
import torch

model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
model.eval()

# backend options
backend = BackendType.QDQ  # Use QDQ mode for ONNX export
# backend = BackendType.SNPE
# backend = BackendType.PPLW8A16
# backend = BackendType.NNIE
# backend = BackendType.Vitis
# backend = BackendType.ONNX_QNN
# backend = BackendType.PPLCUDA
# backend = BackendType.OPENVINO
# backend = BackendType.Tengine_u8
# backend = BackendType.Tensorrt_NLP

# 1. 插入量化节点（scale/zp 为初始值）
model = prepare_by_platform(model, backend)
print("[INFO] Quantization nodes inserted.")

# 2. 创建一些假数据用于校准（实际应用中应使用真实数据集）
dummy_inputs = [torch.randn(8, 3, 224, 224) for _ in range(10)]  # 批量数据用于统计

# 3. 开启校准模式 切换状态
# 启动校准统计 enable_observer()
# 禁用实际量化 disable_fake_quant()
enable_calibration(model)
print("[INFO] Calibration mode enabled.")

# 4. 执行校准前向传播（收集 min/max 统计信息）
with torch.no_grad():
    for i, dummy_input in enumerate(dummy_inputs):  # 多跑几轮收集更多数据
        # 执行前向传播 -> 收集 min/max 统计
        _ = model(dummy_input)
        if (i + 1) % 5 == 0:
            print(f"[INFO] Calibration progress: {(i+1)*10}%")

print("[INFO] Calibration completed.")

# 5. 开启量化模式 切换状态
# 停止校准统计 disable_observer()
# 开启量化 enable_fake_quant()
# 此时会模型里会执行calculate_qparams，计算 scale/zp
enable_quantization(model)
print("[INFO] Quantization mode enabled.")

# 6. 定义输入 shape 并导出 ONNX
input_shape = {'data': [1, 3, 224, 224]}

# Convert and export to ONNX (convert_deploy will generate the ONNX file)
deploy_model = convert_deploy(model, backend, input_shape)

# ONNX file is saved as mqbench_qmodel.onnx in current directory
onnx_path = 'mqbench_qmodel.onnx'
if os.path.exists(onnx_path):
    print(f"ONNX model exported successfully to: {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
else:
    print("ONNX export failed!")