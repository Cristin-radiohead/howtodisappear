import os, sys, cv2, numpy as np, logging
from ais_bench.infer.interface import InferSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """标准对称 Letterbox，确保正向处理与逆向还原严格互逆"""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def postprocess_xyxy(output, conf_thres=0.25, iou_thres=0.45):
    """针对 [cx, cy, w, h, cls(80)] 格式的解析（cxcywh -> xyxy，模型已内置 sigmoid）"""
    pred = output[0].T  # (8400, 84)

    # cxcywh -> xyxy
    cxcy = pred[:, :2]
    wh = pred[:, 2:4]
    boxes_x1y1x2y2 = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)

    # ✅ 模型输出已含 sigmoid，直接取类别概率，不要再激活
    class_scores = pred[:, 4:]

    # ✅ 这两行必须保留（你上次把它们删掉了）
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = max_scores >= conf_thres
    if not np.any(mask):
        logger.info("无高置信度目标")
        return []

    final_boxes = boxes_x1y1x2y2[mask]
    final_scores = max_scores[mask]
    final_ids = class_ids[mask]

    w = final_boxes[:, 2] - final_boxes[:, 0]
    h = final_boxes[:, 3] - final_boxes[:, 1]
    valid = (w > 1) & (h > 1) & \
            (final_boxes[:, 0] >= 0) & (final_boxes[:, 1] >= 0) & \
            (final_boxes[:, 2] <= 640) & (final_boxes[:, 3] <= 640)

    if not np.any(valid):
        logger.info("所有框均无效")
        return []

    final_boxes = final_boxes[valid]
    final_scores = final_scores[valid]
    final_ids = final_ids[valid]

    indices = cv2.dnn.NMSBoxes(
        final_boxes.tolist(),
        final_scores.tolist(),
        conf_thres, iou_thres
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                'class_id': int(final_ids[i]),
                'score': float(final_scores[i]),
                'box': final_boxes[i].tolist()
            })
    return results


def draw_detections(img, detections):
    if len(detections) == 0:
        logger.warning("未检测到任何目标")
        return img
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det['box']]
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"C{det['class_id']}:{det['score']:.2f}",
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


class YOLO11Detector:
    def __init__(self, model_path, device_id=0):
        self.session = InferSession(device_id, model_path)
        logger.info(f"✅ ais_bench 模型加载成功: {model_path}")

    def preprocess(self, img):
        img_lb, ratio, pad = letterbox(img, (640, 640))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :])
        return tensor, ratio, pad

    def detect(self, img, conf_thres=0.25, iou_thres=0.45):
        tensor, ratio, pad = self.preprocess(img)
        outputs = self.session.infer([tensor])
        raw = outputs[0]

        # ===== 调试代码 =====
        logger.info(f"raw.shape = {raw.shape}, raw.dtype = {raw.dtype}")
        logger.info(f"raw.min = {raw.min():.4f}, raw.max = {raw.max():.4f}, raw.mean = {raw.mean():.4f}")

        print("前4维 每行 min/max/mean:")
        for i in range(4):
            r = raw[0, i, :]
            print(f"  dim{i}: min={r.min():.3f}, max={r.max():.3f}, mean={r.mean():.3f}")

        cls_part = raw[0, 4:, :]
        print(f"类别维: min={cls_part.min():.4f}, max={cls_part.max():.4f}, "
              f"mean={cls_part.mean():.6f}, >0.5 的比例={np.mean(cls_part > 0.5):.4f}")

        logits = cls_part
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        print(f"假设是 logits, sigmoid 后 max_score 统计: "
              f"min={probs.max(axis=0).min():.4f}, max={probs.max(axis=0).max():.4f}, "
              f"mean={probs.max(axis=0).mean():.4f}")
        print(f"假设已是概率, max_score 统计: "
              f"mean={logits.max(axis=0).mean():.4f}, "
              f">0.45 的比例={np.mean(logits.max(axis=0) > 0.45):.4f}")
        # ===== 调试结束 =====

        return postprocess_xyxy(raw, conf_thres, iou_thres), ratio, pad


if __name__ == "__main__":
    MODEL_PATH, IMAGE_PATH, OUTPUT_PATH = "/root/yolo11n.om", "/root/test.jpg", "/root/result.jpg"
    if not os.path.exists(IMAGE_PATH):
        logger.error(f"图片不存在: {IMAGE_PATH}"); sys.exit(1)

    detector = YOLO11Detector(MODEL_PATH)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        logger.error("图片读取失败"); sys.exit(1)

    dets, ratio, (dw, dh) = detector.detect(img, conf_thres=0.45, iou_thres=0.45)
    logger.info(f"检测到 {len(dets)} 个目标")

    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d['box']
        orig_x1 = (x1 - dw) / ratio
        orig_y1 = (y1 - dh) / ratio
        orig_x2 = (x2 - dw) / ratio
        orig_y2 = (y2 - dh) / ratio
        d['box'] = [
            max(0, min(w, orig_x1)),
            max(0, min(h, orig_y1)),
            max(0, min(w, orig_x2)),
            max(0, min(h, orig_y2))
        ]

    cv2.imwrite(OUTPUT_PATH, draw_detections(img, dets))
    logger.info(f"✅ 完成! 结果保存至: {OUTPUT_PATH}")