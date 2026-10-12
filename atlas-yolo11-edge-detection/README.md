# Atlas YOLO11 边缘目标检测

在昇腾 Atlas 200I DK A2 开发板上，部署 YOLO11 目标检测模型，实现：
- **图片上传检测**：浏览器上传图片，NPU 推理，返回带检测框的结果
- **摄像头实时检测**：USB 摄像头实时推理，浏览器 MJPEG 流查看

## 🎬 演示

![图片检测](assets/demo-image.png)
![摄像头检测](assets/demo-camera.png)

## 🏗️ 架构
PC (WSL)                    开发板 (Atlas 200I DK A2)
─────────────────────       ────────────────────────────
YOLO11n 训练
↓
导出 ONNX (opset=12)
↓
ATC 转换 ──────────────→  yolo11n.om
↓
ais_bench 推理
↓
HTTP 服务 (端口 5000)
↓
PC 浏览器访问

```

## 🧰 技术栈

| 环节 | 工具 |
|---|---|
| 模型训练 | Ultralytics YOLO11n |
| 模型转换 | CANN 7.0.RC1 + ATC |
| 开发板推理 | ais_bench (Ascend310B4) |
| Web 服务 | Python 标准库 http.server |
| 前端 | 原生 HTML + JavaScript |

## 🚀 快速开始

### 环境要求

- 昇腾 Atlas 200I DK A2 开发板
- CANN 7.0.RC1（开发板自带）
- USB 摄像头（实时检测用）

### 准备模型

将 `yolo11n.onnx` 传到开发板，执行 ATC 转换：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1

atc --model=/root/yolo11n.onnx \
    --framework=5 \
    --output=/root/yolo11n \
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend310B4 \
    --output_type=FP32 \
    --input_format=NCHW
```

图片上传检测

```bash
python3 src/image_detect.py
```

浏览器访问 http://<开发板IP>:5000，上传图片。

摄像头实时检测

```bash
python3 src/camera_web.py
```

浏览器访问 http://<开发板IP>:5000，查看实时画面。

📊 性能

· 推理耗时：~10.28 ms/帧
· 吞吐量：~97 FPS
· 芯片：Ascend310B4

🐛 踩坑记录

详见 docs/troubleshooting.md

主要坑：

· CANN 版本不匹配导致 OM 输出异常
· YOLO11 ONNX 导出时 simplify=True 破坏模型
· 错误地对已是概率的输出再加 sigmoid
· letterbox 预处理必须与坐标还原严格对应

📁 项目结构

见仓库目录树。

📄 许可证

MIT License

🙏 致谢

· Ultralytics
· 昇腾社区

```

---
