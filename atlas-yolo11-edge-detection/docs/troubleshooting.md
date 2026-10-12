# 踩坑记录

## 一、CANN 版本兼容

**问题**：PC 用 CANN 8.5.0 转的 OM，在开发板 CANN 7.0.RC1 上推理输出异常（全 0.5 或 nan）。

**原因**：OM 运行环境的 CANN 版本必须晚于或等于转换时使用的版本。

**解决**：在开发板上用自带 CANN 7.0.RC1 直接 ATC 转换。

## 二、ONNX 导出参数

**问题**：`simplify=True` 导出的 ONNX，检测分数只有 0.07（正常应 0.4+）。

**解决**：改用 `simplify=False, opset=12, dynamic=False`。

## 三、后处理 sigmoid 陷阱

**问题**：对模型输出再加 sigmoid，所有分数挤到 0.5，argmax 随机输出 cat。

**原因**：YOLO11 的 ONNX/OM 输出**已经包含 sigmoid**，是概率值。

**解决**：直接使用 `pred[:, 4:]`，不要再激活。

## 四、letterbox 预处理

**问题**：用 `cv2.resize` 直接缩放到 640×640，图像变形，检测失败。

**解决**：用 letterbox（保持宽高比，补灰边），坐标还原时 `(x - dw) / ratio`。

## 五、ATC 转换内存不足

**问题**：开发板上 ATC 被 Killed。

**解决**：
- 加大 Swap：`fallocate --length 16G /swapfile`
- 限制并行：`export TE_PARALLEL_COMPILER=1`
- 后台运行：`nohup atc ... &`

## 六、ais_bench 多线程问题

**问题**：HTTP 服务多线程调用 `session.infer()`，报 `ctx is NULL`。

**原因**：NPU 会话不能跨线程使用。

**解决**：推理放在主线程，HTTP 服务只负责发送最新帧。
