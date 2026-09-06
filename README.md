# robot_ex1：笔记本电脑与鼠标检测

本项目是 2026 年暑期机器人实验 1，用于在相机画面中检测 `laptop`（笔记本电脑）和 `mouse`（鼠标），并通过 ROS 2 在 Jetson 上发布检测结果与标注图像。

项目仓库：[https://github.com/lyc-0428/robot_ex1](https://github.com/lyc-0428/robot_ex1)

## 核心思路：先完成 YOLOv8，再用 YOLO26 精益求精

本项目采用两个连续的完整工程阶段，而不是同时维护两个相互竞争的模型：

```text
从零开始 → 完成 YOLOv8s 训练、评估、ROS 2 与 Jetson 部署 → 以完成版为基础重新实现 YOLO26s → 追求更好的最终效果
```

1. **第一阶段从零完成 YOLOv8s 项目。** YOLOv8s 是最初的模型选择，因为它生态成熟、资料充足、行为稳定，适合完成从数据标注、数据划分、模型训练和置信度评估，到 Jetson 上 ROS 2 实时推理的全部工作。第一阶段的成果本身就是一个完整、可运行的项目版本。
2. **第二阶段在已完成的 YOLOv8s 项目上进一步完成 YOLO26s。** 在 YOLOv8s 已经训练成功并部署运行之后，为了精益求精，项目又使用更新的 YOLO26s 重新进行了数据优化、模型训练、评估和 Jetson 部署，以追求更好的检测效果。

因此，仓库中保留两个模型各有明确用途：

- **YOLOv8s：第一阶段的完整实现**，是从零开始完成训练、评估和板端部署的首个可用版本。
- **YOLO26s：第二阶段的改进实现**，是在第一版已经成功的基础上，为追求更高精度而重新完成的版本。

这两个模型体现的是“从零到完整 YOLOv8s”和“从完整 YOLOv8s 到改进 YOLO26s”两个阶段。YOLO11 不属于这条最终技术路线，已从当前版本中删除。

两代模型均已在 Jetson 上完成 ROS 2 实时推理。板端录像显示，YOLOv8s 第一版具有更高的运行帧率和较稳定的常见单目标表现；YOLO26s 改进版能够覆盖多鼠标、不同外观和遮挡等更复杂场景，但当前仍需继续优化笔记本检测的帧间稳定性。详细分析见 [YOLOv8s 与 YOLO26s Jetson 板端实测](docs/benchmark_v8_vs_v26.md)。

## 仓库内容

```text
robot_ex1/
├── configs/
│   ├── yolo26s_openimages_mini1k_session_mixed_3way.yaml
│   ├── yolo26s_openimages_mini1k_session_mixed_3way_smoke.yaml
│   └── yolov8s_finetune.yaml
├── docs/
├── evidence/
├── models/
│   ├── finetuned/
│   │   ├── yolo26s_laptop_mouse_best.pt
│   │   ├── yolo26s_laptop_mouse_manifest.json
│   │   └── yolov8s_laptop_mouse_best.pt
│   ├── pretrained/
│   │   ├── yolo26s.pt
│   │   └── yolov8s_coco.pt
│   └── model_manifest.json
├── reports/
│   ├── yolo26s_finetuned_20260902/
│   └── ...                         # YOLOv8s 第一阶段训练与评估记录
├── ros2_ws/                        # ROS 2 robot_vision 包
├── scripts/                        # 数据、训练和评估脚本
├── requirements-train.txt
└── README.md
```

数据集、训练过程目录和缓存不提交到 Git；需要通过 `scripts/` 中的脚本重新生成，或从实验环境单独复制。

## 最终模型：YOLO26s

| 项目 | 内容 |
|---|---|
| 预训练权重 | `models/pretrained/yolo26s.pt` |
| 最终微调权重 | `models/finetuned/yolo26s_laptop_mouse_best.pt` |
| 模型清单 | `models/finetuned/yolo26s_laptop_mouse_manifest.json` |
| 类别 | `0=laptop`、`1=mouse` |
| 输入尺寸 | `512` |
| 训练轮数上限 | `160` |
| 训练数据 | `openimages_mini1k_session_mixed_3way` |

最终权重 SHA-256：

```text
396e391eed2d8635ff6cb27f7576f96b8c81be0a7bbd7a108b7f485e6d93731c
```

预训练权重 SHA-256：

```text
646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
```

## 环境安装

训练记录使用以下环境：

- PyTorch `2.7.1+cu126`
- CUDA `12.6`
- Ultralytics `8.4.131`

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

训练和评估默认使用 GPU `0`，运行前应确认 CUDA 可用。

## 数据准备

最终 YOLO26s 使用两个来源：

- 自采集并标注的 laptop/mouse 数据。
- Open Images V7 中筛选的 laptop/mouse 数据。

最终数据集目录为：

```text
datasets/openimages_mini1k_session_mixed_3way/
├── data.yaml
├── train/images/                   # 812 张
├── train/labels/
├── valid/images/                   # 241 张
├── valid/labels/
├── test/images/                    # 179 张
└── test/labels/
```

类别顺序必须保持为：

```yaml
names:
  - laptop
  - mouse
nc: 2
```

### 1. 准备自采集数据

将原始 YOLO 格式数据放在：

```text
datasets/laptop_mouse_yolov8/
```

按拍摄会话重新划分，避免同一会话的数据同时进入训练集和验证/测试集：

```bash
python scripts/regroup_laptop_mouse_by_session.py
```

默认输出：

```text
datasets/laptop_mouse_yolov8_session_split/
```

### 2. 准备 Open Images 数据

下载 Open Images 元数据：

```bash
bash scripts/download_openimages_metadata.sh
```

构建 1,000 张图片的 laptop/mouse 数据集：

```bash
python scripts/prepare_openimages_mini1k.py
```

重新划分 Open Images，并与按会话划分的自采集数据合并：

```bash
python scripts/repartition_openimages_mini1k.py
```

脚本会检查图片数量和跨数据集划分的 SHA-256 重复，避免数据泄漏。所有数据准备脚本遇到已存在的输出目录时会拒绝覆盖。

## 训练 YOLO26s

训练入口仍名为 `scripts/train_yolov8.py`，但它已经支持通过参数加载 YOLO26 配置并指定输出文件。

### 配置检查

只检查权重、数据集、类别和 CUDA 环境，不启动训练：

```bash
python scripts/train_yolov8.py \
  --config configs/yolo26s_openimages_mini1k_session_mixed_3way.yaml \
  --output-model models/finetuned/yolo26s_laptop_mouse_best.pt \
  --manifest models/finetuned/yolo26s_laptop_mouse_manifest.json \
  --dry-run
```

### 一轮冒烟测试

```bash
python scripts/train_yolov8.py \
  --config configs/yolo26s_openimages_mini1k_session_mixed_3way_smoke.yaml \
  --output-model .cache/yolo26s_smoke/best.pt \
  --manifest .cache/yolo26s_smoke/manifest.json
```

### 完整训练

```bash
python scripts/train_yolov8.py \
  --config configs/yolo26s_openimages_mini1k_session_mixed_3way.yaml \
  --output-model models/finetuned/yolo26s_laptop_mouse_best.pt \
  --manifest models/finetuned/yolo26s_laptop_mouse_manifest.json
```

恢复中断的训练：

```bash
python scripts/train_yolov8.py \
  --config configs/yolo26s_openimages_mini1k_session_mixed_3way.yaml \
  --output-model models/finetuned/yolo26s_laptop_mouse_best.pt \
  --manifest models/finetuned/yolo26s_laptop_mouse_manifest.json \
  --resume
```

默认训练过程保存在：

```text
runs/yolo26s/openimages_mini1k_session_mixed_3way_20260902/
```

## YOLO26s 评估结果

结果位于 `reports/yolo26s_finetuned_20260902/`。检测框按类别匹配，IoU 阈值为 `0.50`。

| 验证集 | 最佳置信度 | Precision | Recall | F1 | Detection accuracy | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 自采集会话隔离验证集 | 0.18 | 0.7051 | 0.3716 | 0.4867 | 0.3216 | 0.3024 | 0.2824 |
| Open Images + 自采集混合验证集 | 0.18 | 0.8247 | 0.5498 | 0.6598 | 0.4923 | 0.5284 | 0.4522 |

重新评估最终模型：

```bash
python scripts/evaluate_finetuned_confidence.py \
  --model models/finetuned/yolo26s_laptop_mouse_best.pt \
  --dataset datasets/openimages_mini1k_session_mixed_3way \
  --split valid \
  --output reports/yolo26s_finetuned_20260902/mixed_valid
```

可通过 `--thresholds`、`--iou`、`--imgsz`、`--batch`、`--device` 和 `--workers` 调整评估参数。

以上是验证集结果，不应当作独立测试集结果；两个验证集的数据组成也不同，不应直接横向比较。

## YOLOv8s 第一阶段完整实现

YOLOv8s 文件用于复现项目第一阶段的完整实验，并保留与第二阶段进行公平比较的依据：

- 配置：`configs/yolov8s_finetune.yaml`
- 预训练权重：`models/pretrained/yolov8s_coco.pt`
- 微调权重：`models/finetuned/yolov8s_laptop_mouse_best.pt`
- 报告：`reports/` 下名称含 `yolov8s` 的目录

YOLOv8s 微调模型在早期独立测试集上的结果为：Precision `0.9118`、Recall `0.9394`、F1 `0.9254`。该结果使用不同的数据划分，不能与上面的 YOLO26s 验证结果直接比较。

## Jetson 板端实测

两代模型都已在 Jetson 上通过 ROS 2 节点连续运行，日志中未发现推理异常。下面的数据来自两段本地演示录像及其同期日志，用于说明部署表现，不等同于使用同一标注测试集得到的模型精度对比。

| 项目 | YOLOv8s | YOLO26s |
|---|---:|---:|
| 演示录像时长 | 17.90 秒 | 107.59 秒 |
| 录像同期显示 FPS 均值 | 32.78 | 23.58 |
| 录像同期显示 FPS 范围 | 30.42–34.40 | 18.83–26.21 |
| 按日志帧号与时间戳估算的完整管线速度 | 约 25.22 FPS | 约 12.29 FPS |

实测中，第一阶段完成的 YOLOv8s 在常见单目标场景下置信度较高、速度更快。第二阶段完成的 YOLO26s 能同时识别多只不同外观的鼠标，并能处理遮挡和画面边缘目标，体现了在已有成果上继续改进的价值；不过其笔记本检测仍有掉框和置信度波动，完整 ROS 2 管线速度也低于 YOLOv8s。

两次板端演示的目标数量、场景难度和阈值均不相同，因此不能仅根据录像数字判断精度。两个模型在同一标注测试集上的离线 A/B 结果和板端分析已分别纳入实验报告；板端证据、计算方法与局限见 [docs/benchmark_v8_vs_v26.md](docs/benchmark_v8_vs_v26.md)。

## Jetson / ROS 2 部署

ROS 2 包位于 `ros2_ws/src/robot_vision`，订阅：

- `/camera/image_raw` (`sensor_msgs/msg/Image`)

发布：

- `/detections` (`std_msgs/msg/String`)：JSON 检测结果、边界框、置信度和 FPS。
- `/detections/image` (`sensor_msgs/msg/Image`)：带检测框和 FPS 的图像。

安装 ROS 2、`cv_bridge` 和相机驱动后构建：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select robot_vision
source install/setup.bash
```

当前节点源码保留了早期 YOLOv8s 的默认绝对路径，因此部署最终模型时必须显式传入 YOLO26s 权重：

```bash
ros2 run robot_vision yolo_detector --ros-args \
  -p model_path:=/absolute/path/to/models/finetuned/yolo26s_laptop_mouse_best.pt \
  -p laptop_confidence:=0.15 \
  -p mouse_confidence:=0.30 \
  -p image_size:=512 \
  -p device:=0
```

查看标注图像：

```bash
ros2 run image_view image_view --ros-args -r image:=/detections/image
```

详细 ROS 2 说明见 `ros2_ws/README.md`。

## 注意事项

- 最终部署权重是 `yolo26s_laptop_mouse_best.pt`，不要误用预训练的 `yolo26s.pt`。
- 模型权重已保存在 Git 中，但数据集和训练过程目录未保存。
- `models/` 被 `.gitignore` 忽略；已经提交的权重仍会被 Git 跟踪，但新增权重时需要明确确认是否应提交。
- 不要用不同数据划分上的指标直接比较 YOLOv8s 与 YOLO26s。

## License

本项目使用 MIT License，见 `LICENSE`。
