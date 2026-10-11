from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练模型
    model = YOLO("yolo11n.pt")  # 会自动下载

    # 开始训练
    model.train(
        data="coco8.yaml",      # 先用官方小数据集跑通
        epochs=3,               # 3轮足够验证流程
        imgsz=640,
        batch=16,               # 显存不够就改小，或设为 -1 自动适配
        device=0                # 有GPU写0；无GPU写 "cpu"
    )