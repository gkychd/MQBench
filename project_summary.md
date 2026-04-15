# MQBench 项目解析

在线文档[https://mqbench.readthedocs.io](https://mqbench.readthedocs.io/)

**MQBench** 是一个基于 PyTorch FX 的开源模型量化工具包（Model Quantization Benchmark），支持 PyTorch 2.5.1。

## 项目目标

- **SOTA 算法**：提供最新的量化算法，支持硬件厂商和研究人员使用学术前沿的量化方法
- **强大工具包**：自动将量化节点插入原始 PyTorch 模块，训练后可转换为目标硬件推理格式

## 目录结构

```
MQBench/
├── mqbench/                    # 主包
│   ├── prepare_by_platform.py   # 量化准备（按平台）
│   ├── advanced_ptq.py       # 高级 PTQ 算法
│   ├── observer.py           # 观察器（收集统计信息）
│   ├── fusion_method.py     # 融合方法
│   ├── convert_deploy.py   # 部署转换
│   ├── weight_equalization.py # 权重均衡化
│   ├── fake_quantize/     # 伪量化算子
│   ├── deploy/             # 部署后端
│   ├── nn/                # 自定义量化模块
│   ├── quantization/      # 量化配置
│   ├── mix_precision/   # 混合精度
│   └── utils/            # 工具函数
├── application/            # 应用示例
│   ├── cls_example/       # 分类任务示例
│   └── yolov5_example/   # YOLOv5 示例
├── test/                   # 测试
└── docs/                  # 文档
```

## 核心模块

### fake_quantize/ - 伪量化算子

| 算法     | 文件                      | 描述                             |
| -------- | ------------------------- | -------------------------------- |
| LSQ      | `lsq.py`                | Learned Step Size Quantization   |
| DSQ      | `dsq.py`                | Differentiable Soft Quantization |
| TQT      | `tqt.py`                | Tensor Quantization Training     |
| ADAROUND | `adaround_quantizer.py` | Adaptive Rounding                |
| QDrop    | `qdrop_quantizer.py`    | 结合 QAT 与 Dropout              |
| PACT     | `pact.py`               | Parameterized Clipping           |
| DoReFa   | `dorefa.py`             | DoReFa Quantization              |
| Fixed    | `fixed.py`              | Fixed-Point Quantization         |

### deploy/ - 部署后端

| 后端         | 文件                       | 描述              |
| ------------ | -------------------------- | ----------------- |
| TensorRT     | `convert_xir.py`         | NVIDIA TensorRT   |
| ONNX QLinear | `deploy_onnx_qlinear.py` | ONNX QLinear 算子 |
| ONNX QNN     | `deploy_onnx_qnn.py`     | ONNX QNN          |
| OpenVINO     | `deploy_openvino.py`     | Intel OpenVINO    |
| SPU          | `deploy_stpu.py`         | SPU 后端          |
| Tengine      | `deploy_tengine.py`      | Tengine 后端      |
| NNIE         | `deploy_nnie.py`         | NNIE 后端         |

## 量化流程

1. **模型准备** (`prepare_by_platform.py`)：根据目标平台插入量化算子
2. **PTQ/QAT 训练**：使用伪量化进行训练
3. **部署转换** (`convert_deploy.py`)：转换为目标硬件格式
4. **推理部署**：在目标设备上运行

## 安装

```bash
git clone git@github.com:ModelTC/MQBench.git
cd MQBench
pip install -v -e .
```

## 参考

- 论文：MQBench: Towards Reproducible and Deployable Model Quantization Benchmark (NeurIPS 2021)
- 文档：https://mqbench.readthedocs.io/en/latest/
