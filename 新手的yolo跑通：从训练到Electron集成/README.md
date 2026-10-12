# YOLO 实时目标检测桌面应用

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4-green)](https://ultralytics.com/)
[![Electron](https://img.shields.io/badge/Electron-33-9cf)](https://www.electronjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

> 从零完成 YOLO 训练、实时推理、Web 接入与 Electron 桌面应用集成。  
> 硬件：NVIDIA RTX 5060 Laptop GPU，Windows。

---

## 🎬 演示

![演示 GIF](assets/demo.gif)

*摄像头实时检测，Electron 窗口显示检测结果。*

---

## ✨ 功能特性

- [x] 使用 Ultralytics YOLO11n 在 COCO8 上完成训练
- [x] 导出 `best.pt` 模型
- [x] Python + OpenCV 实时摄像头推理
- [x] FastAPI 后端提供 `/detect` 接口
- [x] Web 前端调用摄像头并显示检测结果
- [x] Electron 桌面应用内嵌，自动启动 Python 后端
- [x] 跨平台（Windows 验证通过）

---

## 🏗️ 架构

```mermaid
graph LR
    A[摄像头] --> B[前端 HTML/Electron]
    B -->|HTTP POST /detect| C[FastAPI 后端]
    C -->|YOLO 推理| D[best.pt]
    D -->|检测结果| C
    C -->|JSON| B
    B -->|画框显示| A
