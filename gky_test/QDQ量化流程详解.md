# MQBench QDQ 量化流程详解

## 概述

本文档详细介绍使用 MQBench 进行 QDQ（Quantize-Dequantize）量化模型的完整流程，包括量化节点插入、校准、参数计算和 ONNX 导出。

---

## 完整代码示例

```python
import torchvision.models as models
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType
from mqbench.utils.state import enable_calibration
from mqbench.utils.state import enable_quantization
from mqbench.convert_deploy import convert_deploy
import torch
import os

# 1. 加载预训练模型
model = models.__dict__["resnet18"](pretrained=True)
model.eval()

# 2. 选择后端（QDQ 模式适合 ONNX 导出）
backend = BackendType.QDQ

# 3. 插入量化节点
model = prepare_by_platform(model, backend)

# 4. 创建校准数据（多批不同数据）
dummy_inputs = [torch.randn(8, 3, 224, 224) for _ in range(10)]

# 5. 开启校准模式
enable_calibration(model)

# 6. 执行校准前向传播（收集 min/max 统计）
with torch.no_grad():
    for i, dummy_input in enumerate(dummy_inputs):
        _ = model(dummy_input)

# 7. 开启量化模式（计算 scale 和 zero_point）
enable_quantization(model)

# 8. 导出 ONNX
input_shape = {'data': [1, 3, 224, 224]}
deploy_model = convert_deploy(model, backend, input_shape)

# 检查输出文件
onnx_path = 'mqbench_qmodel.onnx'
if os.path.exists(onnx_path):
    print(f"ONNX 导出成功: {onnx_path}, 大小: {os.path.getsize(onnx_path)/1024/1024:.2f} MB")
```

---

## 核心流程

### 流程图

```
用户代码
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ prepare_by_platform(model, backend)                        │
├─────────────────────────────────────────────────────────────┤
│ 1. get_qconfig_by_platform() - 获取量化配置                  │
│ 2. symbolic_trace() - 追踪模型生成 FX 图                     │
│ 3. ModelQuantizer.prepare() - 插入量化节点                   │
│    ├── _fuse_fx() - 融合 Conv+BN+ReLU                       │
│    ├── _weight_quant() - 替换为 QAT 模块（权重量化）          │
│    └── _insert_fake_quantize_for_act_quant() - 插入激活量化 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ enable_calibration(model)                                    │
│ - 遍历所有 FakeQuantize 模块                                 │
│ - enable_observer() - 开启统计收集                           │
│ - disable_fake_quant() - 关闭实际量化                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 前向传播 (model(input))                                     │
│ - Observer.forward() - 统计 min/max 值                      │
│ - 更新 self.min_val 和 self.max_val                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ enable_quantization(model)                                  │
│ - enable_fake_quant() - 开启量化                            │
│ - disable_observer() - 关闭统计                            │
│ - calculate_qparams() - 计算 scale 和 zero_point            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ convert_deploy(model, backend, input_shape)                │
│ - convert_merge_bn() - 融合 BatchNorm                      │
│ - convert_onnx() - 导出 ONNX 模型                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 详细组件说明

### 1. prepare_by_platform()

**位置**: `mqbench/prepare_by_platform.py`

**作用**: 为主流后端准备量化模型

```python
def prepare_by_platform(model, deploy_backend, ...):
    # 获取量化配置
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict)

    # 获取量化器
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](extra_quantizer_dict, extra_fuse_dict)

    # 执行量化准备
    prepared = quantizer.prepare(graph_module, qconfig, is_qat, backend_config, freeze_bn)

    return prepared
```

### 2. ModelQuantizer

**位置**: `mqbench/custom_quantizer/model_quantizer.py`

**注册的量化器**: Tensorrt, NNIE, QDQ

#### prepare() 方法

```python
def prepare(self, model, qconfig, is_qat, backend_config, freeze_bn):
    # 步骤1: 融合模块
    model = _fuse_fx(model, is_qat, self.extra_fuse_dict, backend_config)

    # 步骤2: 权重量化
    model = self._weight_quant(model, qconfig, backend_config, freeze_bn)

    # 步骤3: 激活量化
    model = self._insert_fake_quantize_for_act_quant(model, qconfig)

    return model
```

#### 步骤1: 模块融合 (_fuse_fx)

| 融合前 | 融合后 |
|--------|--------|
| Conv2d → BatchNorm2d → ReLU | ConvBnReLU2d |

#### 步骤2: 权重量化 (_weight_quant)

将普通 Module 替换为 QAT Module：

| 原模块 | QAT 模块 |
|--------|----------|
| nn.Conv2d | nn.qat.modules.conv.Conv2d |
| nn.Linear | nn.qat.modules.linear.Linear |
| nn.ConvBnReLU2d | nn.intrinsic.qat.modules.ConvBnReLU2d |

QAT 模块内部包含 `weight_fake_quantizer`，在前向传播时对权重进行量化。

#### 步骤3: 激活量化 (_insert_fake_quantize_for_act_quant)

在指定层输出后插入 `activation_fake_quantizer`：

```python
# 需要量化激活的层类型
module_type_to_quant_input = (
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.BatchNorm2d,
    torch.nn.Upsample,
    ...
)
```

### 3. 量化节点结构

```
原始:  Input → Conv2d → BN → ReLU → Output

量化后:
Input → activation_fake_quantizer → Conv2d(带weight_fake_quantizer) → activation_fake_quantizer → output
       ↑                                    ↑
    激活量化节点                       权重量化节点（在 Conv 内部）
```

| 节点类型 | 位置 | 作用 |
|----------|------|------|
| weight_fake_quantizer | Conv/Linear 内部 | 量化权重 |
| activation_fake_quantizer | 层与层之间 | 量化激活值 |

### 4. scale 和 zero_point 的计算

#### 计算位置

1. **Observer** (`mqbench/observer.py`) - 统计 min/max
2. **FakeQuantize** (`mqbench/fake_quantize/`) - 计算 scale/zp

#### 计算流程

```python
# observer.py - MinMaxObserver
def forward(self, x):
    # 统计当前批次的 min/max
    min_val_cur, max_val_cur = torch.aminmax(x)

    # 更新全局 min/max
    self.min_val = torch.min(self.min_val, min_val_cur)
    self.max_val = torch.max(self.max_val, max_val_cur)

# quantize_base.py - calculate_qparams
def calculate_qparams(self):
    return self.activation_post_process.calculate_qparams()

# 计算公式（PyTorch）
scale = (max_val - min_val) / (quant_max - quant_min)
zero_point = -min_val / scale
```

#### 各阶段状态

| 阶段 | scale | zero_point | 说明 |
|------|-------|------------|------|
| prepare_by_platform 后 | 1.0 | 0 | 初始值 |
| enable_calibration + 前向传播 | 统计中 | 统计中 | 收集 min/max |
| enable_quantization 后 | 实际值 | 实际值 | 计算得出 |

### 5. 状态切换函数

**位置**: `mqbench/utils/state.py`

```python
def enable_calibration(model):
    """开启校准模式"""
    for submodule in model.modules():
        if isinstance(submodule, FakeQuantizeBase):
            submodule.enable_observer()    # 开启统计
            submodule.disable_fake_quant() # 关闭量化

def enable_quantization(model):
    """开启量化模式"""
    for submodule in model.modules():
        if isinstance(submodule, FakeQuantizeBase):
            submodule.enable_fake_quant()  # 开启量化
            submodule.disable_observer()  # 关闭统计
```

### 6. convert_deploy

**位置**: `mqbench/convert_deploy.py`

```python
def convert_deploy(model, backend_type, input_shape_dict, ...):
    # BACKEND_DEPLOY_FUNCTION[backend_type] 存储了需要执行的函数列表

    for convert_function in BACKEND_DEPLOY_FUNCTION[backend_type]:
        convert_function(deploy_model, **kwargs)
    return deploy_model
```

对于 QDQ 后端，执行顺序：

1. **convert_merge_bn** - 融合 BatchNorm
2. **convert_onnx** - 导出 ONNX 模型

---

## ONNX 导出相关修复

### 问题1: PyTorch 2.11 兼容性问题

修复了以下文件的导入问题：

- `mqbench/fake_quantize/dsq.py` - `_type_utils` 导入
- `mqbench/fake_quantize/lsq.py` - `_type_utils` 导入
- `mqbench/fake_quantize/tqt.py` - `_type_utils` 导入和参数问题
- `mqbench/custom_symbolic_opset.py` - 使用新的 `register_custom_op_symbolic` API
- `mqbench/convert_deploy.py` - 添加 `model.eval()` 和正确的导出参数

### 问题2: ONNX 导出参数

```python
torch.onnx.export(model, dummy_input, onnx_model_path,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=opset_version,
                  do_constant_folding=True,
                  custom_opsets={'' : opset_version},
                  dynamo=False,  # PyTorch 2.11 需要关闭 dynamo
                  training=TrainingMode.EVAL,
                  operator_export_type=OperatorExportTypes.ONNX)
```

---

## 参考资料

- MQBench GitHub: https://github.com/ModelTC/MQBench
- PyTorch 量化文档: https://pytorch.org/docs/stable/quantization.html
- ONNX 导出: https://pytorch.org/docs/stable/onnx_torchscript.html