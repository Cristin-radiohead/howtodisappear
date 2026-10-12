import cv2
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from ais_bench.infer.interface import InferSession

print("Loading model...")
session = InferSession(device_id=0, model_path="/root/yolo11n.om")
print("Model loaded")

latest_frame = None
frame_lock = threading.Lock()

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball",
    33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
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


def postprocess(output, conf_thres=0.45, iou_thres=0.45):
    pred = output[0].T
    cxcy = pred[:, :2]
    wh = pred[:, 2:4]
    boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)

    # 直接取类别分数，不做 sigmoid
    class_scores = pred[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = max_scores >= conf_thres
    if not np.any(mask):
        return []

    final_boxes = boxes[mask]
    final_scores = max_scores[mask]
    final_ids = class_ids[mask]

    w = final_boxes[:, 2] - final_boxes[:, 0]
    h = final_boxes[:, 3] - final_boxes[:, 1]
    valid = (w > 1) & (h > 1) & \
            (final_boxes[:, 0] >= 0) & (final_boxes[:, 1] >= 0) & \
            (final_boxes[:, 2] <= 640) & (final_boxes[:, 3] <= 640)

    if not np.any(valid):
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


def detect(img):
    img_lb, ratio, (dw, dh) = letterbox(img, (640, 640))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :])
    outputs = session.infer([tensor])
    dets = postprocess(outputs[0])

    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d['box']
        d['box'] = [
            max(0, min(w, (x1 - dw) / ratio)),
            max(0, min(h, (y1 - dh) / ratio)),
            max(0, min(w, (x2 - dw) / ratio)),
            max(0, min(h, (y2 - dh) / ratio)),
        ]
    return dets


cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Camera open failed")
    exit()

print("Camera opened")


def inference_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        dets = detect(frame)

        if dets:
            print("Detected:", [(COCO_NAMES.get(d['class_id'], d['class_id']),
                                 round(d['score'], 2)) for d in dets])

        counts = {}
        for d in dets:
            name = COCO_NAMES.get(d['class_id'], 'cls%d' % d['class_id'])
            counts[name] = counts.get(name, 0) + 1
            x1, y1, x2, y2 = [int(round(v)) for v in d['box']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, '%s %.2f' % (name, d['score']),
                        (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset = 30
        for name, count in counts.items():
            cv2.putText(frame, '%s: %d' % (name, count), (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            with frame_lock:
                latest_frame = jpeg.tobytes()


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    data = latest_frame
                if data is None:
                    continue
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
        except Exception as e:
            print("Connection closed:", e)

    def log_message(self, format, *args):
        pass


def start_server():
    server = HTTPServer(('0.0.0.0', 5000), MJPEGHandler)
    server.serve_forever()


if __name__ == '__main__':
    print("Server started: http://0.0.0.0:5000")
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    inference_loop()
